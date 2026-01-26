use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// Types and methods imported from polars::prelude::* used in this plugin:
// - Series: The fundamental Polars data structure (column of data)
// - PolarsResult<T>: Return type for Polars operations (Result<T, PolarsError>)
// - PolarsError: Error type for Polars operations
// - DataType: Enum representing Polars data types (we use DataType::Float64)
// - Float64Chunked: A ChunkedArray specialized for f64 values
// - ListChunked: A ChunkedArray specialized for List values
// - Series methods: .list(), .cast(), .f64(), .clone(), .dtype(), .len()
// - ChunkedArray methods: .into_iter(), .get(), .len()

/// LOWESS (Locally Weighted Scatterplot Smoothing) plugin for Polars.
///
/// Implements the Cleveland (1979) LOWESS algorithm with robustness iterations.
///
/// This plugin handles two modes:
/// 1. Groupby mode: inputs are List columns, compute smoothing per group (parallel)
/// 2. Direct mode: inputs are regular Series, compute single smoothing
///
/// Parameters:
/// - y: Values to smooth
/// - x: Independent variable (e.g., timestamp)
/// - frac_or_window: If positive, interpreted as frac (fraction of data);
///                   if negative, interpreted as window size (dynamic frac)
/// - it: Number of robustness iterations (typically 2)
///
/// The `#[polars_expr(...)]` macro registers this function as a Polars plugin.
#[polars_expr(output_type=Float64)]
fn lowess(inputs: &[Series]) -> PolarsResult<Series> {
    // Validate we received exactly 4 inputs
    if inputs.len() != 4 {
        return Err(PolarsError::ComputeError(
            "lowess expects 4 inputs: y, x, frac_or_window, it".into(),
        ));
    }

    // ===== SIMPLIFIED: Single Series Mode =====
    // Let Polars handle parallelization across groups
    // Our plugin just processes one Series at a time
    let frac_param = inputs[2].cast(&DataType::Float64)?.f64()?.get(0).unwrap_or(0.3);
    let it = inputs[3].cast(&DataType::UInt32)?.u32()?.get(0).unwrap_or(2) as usize;

    match process_group(&inputs[0], &inputs[1], frac_param, it) {
        Some(result) => Ok(result.into_series()),
        None => {
            // Return nulls if processing failed
            let len = inputs[0].len();
            Ok(Series::new("".into(), vec![None::<f64>; len]))
        }
    }
}

/// Process a single group: extract data, sort, apply LOWESS, unsort
///
/// This function handles the data preparation, sorting, and result mapping
/// to ensure the output matches the input order (as scipy does).
fn process_group(y_series: &Series, x_series: &Series, frac_param: f64, it: usize) -> Option<Series> {
    // Cast both series to Float64
    let y_cast = y_series.cast(&DataType::Float64).ok()?;
    let x_cast = x_series.cast(&DataType::Float64).ok()?;
    let y_ca = y_cast.f64().ok()?;
    let x_ca = x_cast.f64().ok()?;

    // Extract (x, y, original_index) tuples, filtering out nulls and non-finite values
    // Rust syntax: Vec::new() creates an empty vector with type inference
    let mut data: Vec<(f64, f64, usize)> = Vec::new();
    for (i, (y_opt, x_opt)) in y_ca.into_iter().zip(x_ca.into_iter()).enumerate() {
        // Pattern matching to extract both values if both are Some
        if let (Some(y), Some(x)) = (y_opt, x_opt) {
            // Only include finite values (not NaN or Inf)
            if x.is_finite() && y.is_finite() {
                data.push((x, y, i));
            }
        }
    }

    // Handle edge case: not enough data points
    if data.len() < 2 {
        return None;
    }

    // Sort by x values (required for LOWESS algorithm)
    // Rust syntax: closures are |args| expression
    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Extract sorted x and y arrays
    let x_sorted: Vec<f64> = data.iter().map(|&(x, _, _)| x).collect();
    let y_sorted: Vec<f64> = data.iter().map(|&(_, y, _)| y).collect();

    // Compute actual frac parameter
    // If frac_param < 0, it represents window size (dynamic frac computation)
    let frac = if frac_param < 0.0 {
        // Dynamic mode: frac = window / (x_max - x_min)
        let window = -frac_param;
        let x_range = x_sorted[x_sorted.len() - 1] - x_sorted[0];
        if x_range > 0.0 {
            (window / x_range).min(1.0).max(0.01)
        } else {
            0.3 // Fallback if all x values are identical
        }
    } else {
        // Direct mode: use frac as-is
        frac_param.min(1.0).max(0.01)
    };

    // Apply LOWESS algorithm
    let y_smooth_sorted = lowess_impl(&y_sorted, &x_sorted, frac, it);

    // Unsort: map smoothed values back to original indices
    let mut y_smooth = vec![f64::NAN; y_ca.len()];
    for (idx, &(_, _, orig_idx)) in data.iter().enumerate() {
        y_smooth[orig_idx] = y_smooth_sorted[idx];
    }

    // Return as Series (transfer ownership to avoid copy)
    Some(Series::new("".into(), y_smooth))
}

/// Core LOWESS algorithm implementation (OPTIMIZED)
///
/// Implements Cleveland's LOWESS with robustness iterations using
/// an efficient sliding window approach for pre-sorted data.
///
/// Key optimization: For univariate LOWESS, once data is sorted by x,
/// the k nearest neighbors are simply the k surrounding points in the
/// sorted array. This reduces complexity from O(n²×log n) to O(n×k).
///
/// Arguments:
/// - y: Values to smooth (sorted by x)
/// - x: Independent variable values (sorted)
/// - frac: Fraction of data to use for each local regression
/// - it: Number of robustness iterations
///
/// Returns: Smoothed y values
fn lowess_impl(y: &[f64], x: &[f64], frac: f64, it: usize) -> Vec<f64> {
    let n = x.len();
    // k = number of nearest neighbors to use
    let k = ((frac * n as f64).ceil() as usize).max(2).min(n);

    // Pre-allocate output and working arrays
    let mut y_smooth: Vec<f64> = vec![0.0; n];
    let mut robustness_weights: Vec<f64> = vec![1.0; n];
    let mut residuals: Vec<f64> = vec![0.0; n];

    // Main iteration loop: 0 = initial fit, 1..=it = robustness iterations
    for iter in 0..=it {
        // For each point, perform local weighted regression
        for i in 0..n {
            // OPTIMIZED: Find k nearest neighbors in sorted 1D data
            // For sorted data, we expand left/right from point i until we have k neighbors

            // Find initial window centered at i
            let half_k: usize = k / 2;
            let mut left: usize = i.saturating_sub(half_k);
            let mut right: usize = (i + half_k).min(n - 1);

            // Expand to ensure we have at least k points
            if right - left + 1 < k {
                if right == n - 1 {
                    left = n.saturating_sub(k);
                } else if left == 0 {
                    right = (k - 1).min(n - 1);
                } else {
                    right = left + k - 1;
                }
            }

            // Now find exactly k nearest by comparing distances at boundaries
            // and shrinking the window from the farther side
            while right - left + 1 > k {
                let left_dist: f64 = (x[left] - x[i]).abs();
                let right_dist: f64 = (x[right] - x[i]).abs();

                if left_dist > right_dist {
                    left += 1;
                } else {
                    right -= 1;
                }
            }

            // Compute h = max distance to any point in the k-neighborhood
            let left_dist: f64 = (x[left] - x[i]).abs();
            let right_dist: f64 = (x[right] - x[i]).abs();
            let h: f64 = left_dist.max(right_dist);

            // Avoid division by zero if all points are identical
            if h == 0.0 {
                y_smooth[i] = y[i];
                continue;
            }

            // Build local data and weights for regression from window [left, right]
            let mut x_local: Vec<f64> = Vec::with_capacity(k);
            let mut y_local: Vec<f64> = Vec::with_capacity(k);
            let mut weights_local: Vec<f64> = Vec::with_capacity(k);

            for j in left..=right {
                let dist: f64 = (x[j] - x[i]).abs();
                // Tricube weight based on normalized distance
                let u: f64 = dist / h;
                let tricube_w: f64 = tricube_weight(u);
                // Combined weight = tricube × robustness weight
                let combined_w: f64 = tricube_w * robustness_weights[j];

                if combined_w > 0.0 {
                    x_local.push(x[j]);
                    y_local.push(y[j]);
                    weights_local.push(combined_w);
                }
            }

            // Perform weighted linear regression
            if weights_local.is_empty() {
                y_smooth[i] = y[i]; // Fallback
            } else {
                let (beta0, beta1) = local_linear_regression(&x_local, &y_local, &weights_local);
                // Predict smoothed value at x[i]
                y_smooth[i] = beta0 + beta1 * x[i];
            }
        }

        // Update robustness weights for next iteration
        if iter < it {
            // Compute residuals
            for i in 0..n {
                residuals[i] = (y[i] - y_smooth[i]).abs();
            }
            // Compute new robustness weights based on residuals
            robustness_weights = compute_robustness_weights(&residuals);
        }
    }

    y_smooth
}

/// Tricube weight function: (1 - |u|³)³ for |u| < 1, else 0
///
/// This is the standard LOWESS proximity weight function.
/// Points closer to the target point get higher weight.
///
/// Rust syntax: `powi(n)` raises to integer power n
fn tricube_weight(u: f64) -> f64 {
    if u.abs() >= 1.0 {
        0.0
    } else {
        let tmp = 1.0 - u.abs().powi(3);
        tmp.powi(3)
    }
}

/// Bisquare weight function: (1 - u²)² for |u| < 1, else 0
///
/// This is used for robustness iterations to downweight outliers.
fn bisquare_weight(u: f64) -> f64 {
    if u.abs() >= 1.0 {
        0.0
    } else {
        let tmp = 1.0 - u * u;
        tmp * tmp
    }
}

/// Compute robustness weights from residuals using bisquare function
///
/// Weights are based on: w_i = bisquare(r_i / (6 × median(r)))
/// The factor of 6 is standard in robust statistics.
fn compute_robustness_weights(residuals: &[f64]) -> Vec<f64> {
    // Compute median absolute residual
    let mut sorted_residuals = residuals.to_vec();
    sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_residuals.len();
    let median = if n == 0 {
        0.0
    } else if n % 2 == 0 {
        (sorted_residuals[n / 2 - 1] + sorted_residuals[n / 2]) / 2.0
    } else {
        sorted_residuals[n / 2]
    };

    // Avoid division by zero
    if median == 0.0 {
        return vec![1.0; n];
    }

    // Compute bisquare weights
    // Rust syntax: `.iter().map(...).collect()` transforms each element
    residuals
        .iter()
        .map(|&r| {
            let u = r / (6.0 * median);
            bisquare_weight(u)
        })
        .collect()
}

/// Weighted linear regression: y = β₀ + β₁x
///
/// Solves the weighted least squares problem using normal equations:
/// [sum_w      sum_wx ] [β₀]   [sum_wy ]
/// [sum_wx     sum_wxx] [β₁] = [sum_wxy]
///
/// Returns: (intercept, slope)
fn local_linear_regression(x: &[f64], y: &[f64], weights: &[f64]) -> (f64, f64) {
    let n = x.len();

    // Compute weighted sums
    // Rust syntax: `mut` allows variable to be modified
    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxx = 0.0;
    let mut sum_wxy = 0.0;

    for i in 0..n {
        if weights[i] > 0.0 && x[i].is_finite() && y[i].is_finite() {
            sum_w += weights[i];
            sum_wx += weights[i] * x[i];
            sum_wy += weights[i] * y[i];
            sum_wxx += weights[i] * x[i] * x[i];
            sum_wxy += weights[i] * x[i] * y[i];
        }
    }

    // Solve normal equations
    let denom = sum_w * sum_wxx - sum_wx * sum_wx;

    if denom.abs() < 1e-10 || sum_w == 0.0 {
        // Degenerate case: return weighted mean (horizontal line)
        let beta0 = sum_wy / sum_w;
        return (beta0, 0.0);
    }

    let beta1 = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
    let beta0 = (sum_wy - beta1 * sum_wx) / sum_w;

    (beta0, beta1)
}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tricube_weight() {
        // Test boundary conditions
        assert_eq!(tricube_weight(0.0), 1.0);
        assert_eq!(tricube_weight(1.0), 0.0);
        assert_eq!(tricube_weight(-1.0), 0.0);
        assert_eq!(tricube_weight(1.5), 0.0);

        // Test intermediate value
        let w = tricube_weight(0.5);
        assert!((w - 0.669921875).abs() < 1e-6);
    }

    #[test]
    fn test_bisquare_weight() {
        assert_eq!(bisquare_weight(0.0), 1.0);
        assert_eq!(bisquare_weight(1.0), 0.0);
        assert_eq!(bisquare_weight(-1.0), 0.0);

        let w = bisquare_weight(0.5);
        assert!((w - 0.5625).abs() < 1e-6);
    }

    #[test]
    fn test_local_linear_regression_simple() {
        // Test on simple linear data: y = 2x
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let w = vec![1.0, 1.0, 1.0];

        let (b0, b1) = local_linear_regression(&x, &y, &w);
        assert!((b0 - 0.0).abs() < 1e-6);
        assert!((b1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_lowess_linear_data() {
        // LOWESS on perfect linear data should return approximately the same data
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = lowess_impl(&y, &x, 0.6, 2);

        // Results should be close to original (within tolerance for smoothing)
        for (yi, ri) in y.iter().zip(result.iter()) {
            assert!((yi - ri).abs() < 0.5);
        }
    }
}

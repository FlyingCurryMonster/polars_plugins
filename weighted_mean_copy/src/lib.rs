use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// Types and methods imported from polars::prelude::* used in this plugin:
// - Series: The fundamental Polars data structure (column of data)
// - PolarsResult<T>: Return type for Polars operations (Result<T, PolarsError>)
// - PolarsError: Error type for Polars operations
// - DataType: Enum representing Polars data types (we use DataType::Float64)
// - Float64Chunked: A ChunkedArray specialized for f64 values
//   - Methods: .into_iter(), .get(), .len()
// - Series methods: .cast(), .f64(), .new()

/// Computes the weighted mean of values.
///
/// SIMPLIFIED VERSION: Just processes one Series at a time.
/// When used with groupby, Polars calls this function per group and handles parallelization.
///
/// The `#[polars_expr(...)]` macro registers this function as a Polars plugin.
/// `output_type = Float64` means the result will be a Float64 column.
#[polars_expr(output_type = Float64)]
fn weighted_mean(inputs: &[Series]) -> PolarsResult<Series> {
    // Validate we received exactly two inputs: values and weights
    if inputs.len() != 2 {
        return Err(PolarsError::ComputeError(
            "weighted_mean expects two inputs: values and weights".into(),
        ));
    }

    // ===== SIMPLIFIED: Single Series Mode =====
    // Let Polars handle parallelization across groups
    // Our plugin just processes one Series at a time

    // Cast inputs to Float64 for numeric operations
    let values = inputs[0].cast(&DataType::Float64)?;
    let weights = inputs[1].cast(&DataType::Float64)?;
    let values_ca = values.f64()?;
    let weights_ca = weights.f64()?;

    // Initialize accumulators for the weighted sum calculation
    let mut sum_w = 0.0_f64;   // sum of weights
    let mut sum_wx = 0.0_f64;  // sum of (weight × value)

    // Handle two cases: 
    // scalar weight broadcasts to array of weights
    // vector weights stay the same

    // compute the weighted sum
    if weights_ca.len() == 1 {
        let w_vec: Vec<Option<f64>> = weights_ca.into_iter().collect();
    } else {
        let w_vec: Vec<Option<f64>> = weights_ca.into_iter().collect();
    }

    if weights_ca.len() == 1 {
        // Scalar weight case: broadcast the single weight across all values
        if let Some(w) = weights_ca.get(0) {
            if w.is_finite() {
                for v_opt in values_ca.into_iter() {
                    if let Some(v) = v_opt {
                        if v.is_finite() {
                            sum_w += w;
                            sum_wx += w * v;
                        }
                    }
                }
            }
        }
    } else {
        // Array weight case: each value has its own corresponding weight
        for (v_opt, w_opt) in values_ca.into_iter().zip(weights_ca) {
            if let (Some(v), Some(w)) = (v_opt, w_opt) {
                if v.is_finite() && w.is_finite() {
                    sum_w += w;
                    sum_wx += w * v;
                }
            }
        }
    }

    // Compute the weighted mean (or None if no valid values)
    let result = if sum_w == 0.0 {
        None
    } else {
        Some(sum_wx / sum_w)
    };

    // Return as a single-element Series
    Ok(Series::new("".into(), vec![result]))
}

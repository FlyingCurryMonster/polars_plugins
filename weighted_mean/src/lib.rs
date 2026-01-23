use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

// Types and methods imported from polars::prelude::* used in this plugin:
// - Series: The fundamental Polars data structure (column of data)
// - PolarsResult<T>: Return type for Polars operations (Result<T, PolarsError>)
// - PolarsError: Error type for Polars operations
// - DataType: Enum representing Polars data types (we use DataType::Float64)
// - Float64Chunked: A ChunkedArray specialized for f64 values
//   - Methods: .apply_amortized_generic(), .into_iter(), .get(), .len()
// - Series methods: .list(), .cast(), .f64(), .clone(), .dtype(), .len(), .into_series()
// - ChunkedArray methods: .into_iter(), .get()

/// Computes the weighted mean of values.
///
/// This plugin handles two modes:
/// 1. Aggregation mode (groupby): inputs are List columns, compute mean per group
/// 2. Direct mode: inputs are regular Series, compute a single weighted mean
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

    // ===== CASE 1: Groupby Aggregation Mode =====
    // When used with groupby, Polars passes data as List columns where each list
    // contains the values for one group. We need to compute one mean per list.
    //
    // Rust syntax note: `if let Ok(value) = expression` is pattern matching.
    // It tries to extract a success value from a Result type.
    if let Ok(values_lc) = inputs[0].list() {
        // Extract the weight value. In groupby mode, weights is typically a scalar
        // (e.g., pl.lit(1.0)), not a list per group.
        let scalar_w = if inputs[1].list().is_ok() {
            // If weights is also a list, just use uniform weight of 1.0
            1.0
        } else {
            // Cast to Float64 and extract the scalar weight value
            // The `?` operator propagates errors up if cast fails
            let w = inputs[1].cast(&DataType::Float64)?;
            // Convert to f64 ChunkedArray, get first element, default to 1.0 if None
            w.f64()?.get(0).unwrap_or(1.0)
        };

        // Process each list (group) and compute its weighted mean
        // `apply_amortized_generic` applies a closure to each list element efficiently
        // The closure returns Option<f64>: Some(mean) or None if the list is empty/invalid
        let out: Float64Chunked = values_lc.apply_amortized_generic(|opt_s| {
            // Extract the Series from Option (returns None if group is null)
            let s = opt_s?;
            // Cast the inner series to Float64 for numeric operations
            let s = s.as_ref().cast(&DataType::Float64).ok()?;
            // Get the Float64 ChunkedArray to iterate over values
            let ca = s.f64().ok()?;

            // Initialize accumulators for weighted sum
            // Rust syntax: explicit type `f64` ensures floating-point math
            let mut sum_w = 0.0_f64;
            let mut sum_wx = 0.0_f64;

            // Iterate over all values in this group
            for v_opt in ca.into_iter() {
                // Pattern match: only process Some(v), skip None (null values)
                if let Some(v) = v_opt {
                    // Skip NaN and infinite values to avoid corrupting the mean
                    if v.is_finite() {
                        sum_w += scalar_w;
                        sum_wx += scalar_w * v;
                    }
                }
            }

            // Return weighted mean, or None if no valid values
            if sum_w == 0.0 { None } else { Some(sum_wx / sum_w) }
        });

        // Convert the Float64Chunked back to a Series and return
        return Ok(out.into_series());
    }

    // ===== CASE 2: Direct (Non-Aggregation) Mode =====
    // When called outside groupby, inputs are regular Series.
    // We compute a single weighted mean across all values.

    // Clone the input series so we can modify them
    // Rust requires explicit cloning; variables are moved by default
    let values = inputs[0].clone();
    let weights = inputs[1].clone();

    // Cast both series to Float64 for numeric operations
    // The `?` propagates any errors that occur during casting
    let values = values.cast(&DataType::Float64)?;
    let weights = weights.cast(&DataType::Float64)?;
    // Extract the underlying Float64ChunkedArray from each Series
    let values = values.f64()?;
    let weights = weights.f64()?;

    // Initialize accumulators for the weighted sum calculation
    let mut sum_w = 0.0_f64;   // sum of weights
    let mut sum_wx = 0.0_f64;  // sum of (weight × value)

    // Handle two cases: scalar weight (broadcast) or array of weights
    //
    // Rust syntax: `mut` keyword allows a variable to be modified after declaration
    if weights.len() == 1 {
        // Scalar weight case: broadcast the single weight across all values
        // Example: weighted_mean_expr(pl.col('Q'), pl.lit(1.0))
        if let Some(w) = weights.get(0) {
            // Only proceed if weight is a finite number (not NaN or infinity)
            if w.is_finite() {
                // Iterate through all values
                for v_opt in values.into_iter() {
                    // Pattern match to extract value from Option (skip nulls)
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
        // Example: weighted_mean_expr(pl.col('Q'), pl.col('uncertainty'))
        //
        // Rust syntax: `.zip()` combines two iterators into pairs
        // Pattern matching `(v_opt, w_opt)` destructures the tuple
        for (v_opt, w_opt) in values.into_iter().zip(weights) {
            // Match on the tuple to extract both value and weight if both are Some
            if let (Some(v), Some(w)) = (v_opt, w_opt) {
                // Only include finite numbers in the calculation
                if v.is_finite() && w.is_finite() {
                    sum_w += w;
                    sum_wx += w * v;
                }
            }
        }
    }

    // Compute the final weighted mean
    // Return None if no valid values were found (avoids division by zero)
    let out = if sum_w == 0.0 {
        None
    } else {
        Some(sum_wx / sum_w)
    };

    // Polars plugins must return a Series, even for scalar results
    // Create a single-element Series containing our result
    // Rust syntax: `&[out]` creates a slice reference to an array
    // The empty string "" means Polars will assign the column name
    Ok(Series::new("", &[out]))
}

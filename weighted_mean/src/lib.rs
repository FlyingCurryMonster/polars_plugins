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
    let values: Series= inputs[0].cast(&DataType::Float64)?;
    let weights: Series= inputs[1].cast(&DataType::Float64)?;
    let values_ca: &ChunkedArray<Float64Type> = values.f64()?;
    let weights_ca: &ChunkedArray<Float64Type> = weights.f64()?;

    // Initialize accumulators for the weighted sum calculation
    let mut sum_w: f64 = 0.0_f64;   // sum of weights
    let mut sum_wx: f64 = 0.0_f64;  // sum of (weight × value)

    // Handle two cases: scalar weight (broadcast) or array of weights
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

#[polars_expr(output_type = Float64)]
fn weighted_mean_2(inputs: &[Series]) -> PolarsResult<Series> {
    // This is just a placeholder to demonstrate multiple functions in the same plugin.
    // In practice, you would implement different logic here.
    if inputs.len() != 2 {
        return Err(PolarsError::ComputeError(
            "weighted_mean_2 expects two inputs: values and weights".into(),
        ));
    }

    let values: Series= inputs[0].cast(&DataType::Float64)?;
    let weights: Series= inputs[1].cast(&DataType::Float64)?;
    let values_ca: &ChunkedArray<Float64Type> = values.f64()?;

    let weights_ca: ChunkedArray<Float64Type> = if weights.len() == 1 {
        // broadcast the single weight into a vector of weights of size ones
        let scalar_weight: f64 = weights.f64()?.get(0)
                                .ok_or_else(|| PolarsError::ComputeError("Failed to get scalar weight".into()))?;
        ChunkedArray::full("ones", scalar_weight, values_ca.len())
    } else{
        weights.f64()?.clone()
    };
    
    let mut weights_sum: f64 = 0.0_f64;
    let mut weighted_sum: f64 = 0.0_f64;

    for (v_opt, w_opt) in values_ca.into_iter().zip(&weights_ca) {
        // define v_i and w_i from the loop
        // add w_i to weights_sum and add w_i * v_i to weighted_sum
        if let (Some(v), Some(w)) = (v_opt, w_opt) {
            if v.is_finite() && w.is_finite() {
                weights_sum += w;
                weighted_sum += w * v;
            }  
        }
    }

    let output: Option<f64> = if weights_sum == 0.0 {
        None
    }
    else {
        Some(weighted_sum / weights_sum)
    };

    Ok(Series::new("".into(), vec![output]))
    }



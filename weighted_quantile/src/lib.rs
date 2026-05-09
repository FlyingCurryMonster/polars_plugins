use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

#[derive(Clone, Debug)]
struct WeightedPoint {
    value: f64,
    weight: f64,
}

/// Weighted quantile using mass-midpoint interpolation.
///
/// Inputs:
/// - values
/// - weights
/// - quantile scalar in [0, 1]
/// - min_samples scalar
#[polars_expr(output_type = Float64)]
fn weighted_quantile(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 4 {
        return Err(PolarsError::ComputeError(
            "weighted_quantile expects inputs: values, weights, quantile, min_samples".into(),
        ));
    }

    let values = inputs[0].cast(&DataType::Float64)?;
    let weights = inputs[1].cast(&DataType::Float64)?;
    let quantile = inputs[2]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    let min_samples = inputs[3]
        .cast(&DataType::UInt32)?
        .u32()?
        .get(0)
        .unwrap_or(1) as usize;

    let value_ca = values.f64()?;
    let weight_ca = weights.f64()?;

    let result = weighted_quantile_impl(value_ca, weight_ca, quantile, min_samples);
    Ok(Series::new("".into(), vec![result]))
}

fn weighted_quantile_impl(
    values: &Float64Chunked,
    weights: &Float64Chunked,
    quantile: f64,
    min_samples: usize,
) -> Option<f64> {
    let mut points: Vec<WeightedPoint> = Vec::with_capacity(values.len());

    if weights.len() == 1 {
        let weight = weights.get(0)?;
        if !weight.is_finite() || weight <= 0.0 {
            return None;
        }

        for value in values.into_iter().flatten() {
            if value.is_finite() {
                points.push(WeightedPoint { value, weight });
            }
        }
    } else {
        for (value_opt, weight_opt) in values.into_iter().zip(weights.into_iter()) {
            if let (Some(value), Some(weight)) = (value_opt, weight_opt) {
                if value.is_finite() && weight.is_finite() && weight > 0.0 {
                    points.push(WeightedPoint { value, weight });
                }
            }
        }
    }

    if points.len() < min_samples {
        return None;
    }

    points.sort_by(|a, b| {
        a.value
            .partial_cmp(&b.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_weight = points.iter().map(|point| point.weight).sum::<f64>();
    if !total_weight.is_finite() || total_weight <= 0.0 {
        return None;
    }

    let target = quantile.clamp(0.0, 1.0) * total_weight;
    let mut cumulative_weight = 0.0_f64;
    let mut lower: Option<(f64, f64)> = None;
    let mut upper: Option<(f64, f64)> = None;

    for point in &points {
        cumulative_weight += point.weight;
        let midpoint = cumulative_weight - point.weight / 2.0;

        if midpoint <= target {
            lower = Some((point.value, midpoint));
        }
        if upper.is_none() && midpoint >= target {
            upper = Some((point.value, midpoint));
        }
    }

    let (x_lower, p_lower) = lower.unwrap_or_else(|| {
        let first = &points[0];
        (first.value, first.weight / 2.0)
    });
    let (x_upper, p_upper) = upper.unwrap_or_else(|| {
        let last = points.last().expect("points is non-empty");
        (last.value, total_weight - last.weight / 2.0)
    });

    if p_lower == p_upper {
        return Some(x_lower);
    }

    let fraction = (target - p_lower) / (p_upper - p_lower);
    Some(x_lower + fraction * (x_upper - x_lower))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ca(values: &[Option<f64>]) -> Float64Chunked {
        Float64Chunked::from_iter(values.iter().copied())
    }

    #[test]
    fn equal_weights_match_midpoint_median() {
        let values = ca(&[Some(0.0), Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        let weights = ca(&[Some(0.2), Some(0.2), Some(0.2), Some(0.2), Some(0.2)]);
        let result = weighted_quantile_impl(&values, &weights, 0.5, 1).unwrap();
        assert!((result - 2.0).abs() < 1e-12);
    }

    #[test]
    fn ignores_null_nan_and_non_positive_weights() {
        let values = ca(&[Some(1.0), None, Some(f64::NAN), Some(5.0), Some(10.0)]);
        let weights = ca(&[Some(1.0), Some(1.0), Some(1.0), Some(0.0), Some(1.0)]);
        let result = weighted_quantile_impl(&values, &weights, 0.5, 2).unwrap();
        assert!((result - 5.5).abs() < 1e-12);
    }

    #[test]
    fn returns_none_below_min_samples() {
        let values = ca(&[Some(1.0), Some(2.0)]);
        let weights = ca(&[Some(1.0), Some(1.0)]);
        assert!(weighted_quantile_impl(&values, &weights, 0.5, 3).is_none());
    }
}

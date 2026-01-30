use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use lowess::prelude::*;

/// Polars plugin entrypoint for LOWESS smoothing.
///
/// Inputs (4):
/// - y
/// - x
/// - frac_or_window
/// - it
///
#[polars_expr(output_type = Float64)]
fn lowess(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 4 {
        return Err(PolarsError::ComputeError(
            "lowess expects 4 inputs: y, x, frac_or_window, it".into(),
        ));
    }

    let frac = inputs[2]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .unwrap_or(0.3)
        .clamp(0.01, 1.0);

    let it = inputs[3]
        .cast(&DataType::UInt32)?
        .u32()?
        .get(0)
        .unwrap_or(2) as usize;

    match process_group(&inputs[0], &inputs[1], frac, it) {
        Some(result) => Ok(result),
        None => Ok(Series::new("".into(), vec![None::<f64>; inputs[0].len()])),
    }
}

fn process_group(y_series: &Series, x_series: &Series, frac: f64, it: usize) -> Option<Series> {
    let y_cast = y_series.cast(&DataType::Float64).ok()?;
    let x_cast = x_series.cast(&DataType::Float64).ok()?;
    let y_ca = y_cast.f64().ok()?;
    let x_ca = x_cast.f64().ok()?;

    let mut data: Vec<(f64, f64, usize)> = Vec::new();
    for (i, (y_opt, x_opt)) in y_ca.into_iter().zip(x_ca.into_iter()).enumerate() {
        if let (Some(y), Some(x)) = (y_opt, x_opt) {
            if x.is_finite() && y.is_finite() {
                data.push((x, y, i));
            }
        }
    }

    if data.len() < 2 {
        return None;
    }

    // Sort by x, tie-breaking on original index for determinism.
    data.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
    });

    let x_sorted: Vec<f64> = data.iter().map(|&(x, _, _)| x).collect();
    let y_sorted: Vec<f64> = data.iter().map(|&(_, y, _)| y).collect();

    let model = Lowess::new()
        .fraction(frac)
        .iterations(it)
        .adapter(Adapter::Batch)
        .build()
        .ok()?;

    let result = model.fit(&x_sorted, &y_sorted).ok()?;
    let y_smooth_sorted = result.y;

    if y_smooth_sorted.len() != data.len() {
        return None;
    }

    let mut y_smooth = vec![f64::NAN; y_ca.len()];
    for (sorted_idx, &(_, _, orig_idx)) in data.iter().enumerate() {
        y_smooth[orig_idx] = y_smooth_sorted[sorted_idx];
    }

    Some(Series::new("".into(), y_smooth))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowess_matches_legacy_approximately() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![2.0, 4.1, 5.9, 8.2, 9.8, 12.2];

        let legacy = lowess_legacy::lowess_impl_legacy(&y, &x, 0.6, 2);

        let model = Lowess::new()
            .fraction(0.6)
            .iterations(2)
            .adapter(Adapter::Batch)
            .build()
            .unwrap();
        let ours = model.fit(&x, &y).unwrap().y;

        assert_eq!(legacy.len(), ours.len());
        for (a, b) in legacy.iter().zip(ours.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

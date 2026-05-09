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


#[derive(Clone, Debug)]
struct RollingPoint {
    value: f64,
    weight: f64,
    row_idx: usize,
    sorted_idx: usize,
}

#[derive(Clone, Debug)]
struct FenwickTree {
    values: Vec<f64>,
}

impl FenwickTree {
    fn new(len: usize) -> Self {
        Self {
            values: vec![0.0; len + 1],
        }
    }

    fn add(&mut self, idx: usize, delta: f64) {
        let mut tree_idx = idx + 1;
        while tree_idx < self.values.len() {
            self.values[tree_idx] += delta;
            tree_idx += tree_idx & (!tree_idx + 1);
        }
    }

    fn prefix_sum(&self, idx: usize) -> f64 {
        let mut tree_idx = idx + 1;
        let mut sum = 0.0;
        while tree_idx > 0 {
            sum += self.values[tree_idx];
            tree_idx -= tree_idx & (!tree_idx + 1);
        }
        sum
    }

    fn total(&self) -> f64 {
        if self.values.len() <= 1 {
            0.0
        } else {
            self.prefix_sum(self.values.len() - 2)
        }
    }

    fn first_prefix_ge(&self, target: f64) -> Option<usize> {
        if self.values.len() <= 1 || target > self.total() {
            return None;
        }
        if target <= 0.0 {
            return Some(0);
        }

        let mut idx = 0_usize;
        let mut bit = 1_usize;
        while bit < self.values.len() {
            bit <<= 1;
        }
        bit >>= 1;

        let mut running = 0.0;
        while bit > 0 {
            let next = idx + bit;
            if next < self.values.len() && running + self.values[next] < target {
                idx = next;
                running += self.values[next];
            }
            bit >>= 1;
        }

        if idx < self.values.len() - 1 {
            Some(idx)
        } else {
            None
        }
    }

    fn first_count_ge(&self, target_count: f64) -> Option<usize> {
        self.first_prefix_ge(target_count)
    }
}

/// Rolling weighted quantile over a full series, intended to be used with `.over(group)`.
///
/// Inputs:
/// - values
/// - weights
/// - window_size scalar
/// - quantile scalar in [0, 1]
/// - min_samples scalar
#[polars_expr(output_type = Float64)]
fn rolling_weighted_quantile(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.len() != 5 {
        return Err(PolarsError::ComputeError(
            "rolling_weighted_quantile expects inputs: values, weights, window_size, quantile, min_samples".into(),
        ));
    }

    let values = inputs[0].cast(&DataType::Float64)?;
    let weights = inputs[1].cast(&DataType::Float64)?;
    let window_size = inputs[2]
        .cast(&DataType::UInt32)?
        .u32()?
        .get(0)
        .unwrap_or(1) as usize;
    let quantile = inputs[3]
        .cast(&DataType::Float64)?
        .f64()?
        .get(0)
        .unwrap_or(0.5)
        .clamp(0.0, 1.0);
    let min_samples = inputs[4]
        .cast(&DataType::UInt32)?
        .u32()?
        .get(0)
        .unwrap_or(1) as usize;

    let value_ca = values.f64()?;
    let weight_ca = weights.f64()?;
    let result = rolling_weighted_quantile_impl(
        value_ca,
        weight_ca,
        window_size.max(1),
        quantile,
        min_samples,
    );
    Ok(Series::new("".into(), result))
}

fn rolling_weighted_quantile_impl(
    values: &Float64Chunked,
    weights: &Float64Chunked,
    window_size: usize,
    quantile: f64,
    min_samples: usize,
) -> Vec<Option<f64>> {
    let len = values.len();
    let mut sorted_points: Vec<RollingPoint> = values
        .into_iter()
        .zip(weights.into_iter())
        .enumerate()
        .filter_map(|(row_idx, (value_opt, weight_opt))| match (value_opt, weight_opt) {
            (Some(value), Some(weight)) if value.is_finite() && weight.is_finite() && weight > 0.0 => {
                Some(RollingPoint {
                    value,
                    weight,
                    row_idx,
                    sorted_idx: 0,
                })
            }
            _ => None,
        })
        .collect();

    sorted_points.sort_by(|a, b| {
        a.value
            .partial_cmp(&b.value)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.row_idx.cmp(&b.row_idx))
    });

    let mut points_by_row: Vec<Option<RollingPoint>> = vec![None; len];
    for (sorted_idx, point) in sorted_points.iter_mut().enumerate() {
        point.sorted_idx = sorted_idx;
        points_by_row[point.row_idx] = Some(point.clone());
    }

    let mut active_weights = vec![0.0_f64; sorted_points.len()];
    let mut weight_tree = FenwickTree::new(sorted_points.len());
    let mut count_tree = FenwickTree::new(sorted_points.len());
    let mut valid_count = 0_usize;
    let mut output = Vec::with_capacity(len);

    for row_idx in 0..len {
        if let Some(point) = &points_by_row[row_idx] {
            active_weights[point.sorted_idx] = point.weight;
            weight_tree.add(point.sorted_idx, point.weight);
            count_tree.add(point.sorted_idx, 1.0);
            valid_count += 1;
        }

        if row_idx >= window_size {
            if let Some(point) = &points_by_row[row_idx - window_size] {
                active_weights[point.sorted_idx] = 0.0;
                weight_tree.add(point.sorted_idx, -point.weight);
                count_tree.add(point.sorted_idx, -1.0);
                valid_count -= 1;
            }
        }

        if valid_count < min_samples {
            output.push(None);
            continue;
        }

        let total_weight = weight_tree.total();
        if !total_weight.is_finite() || total_weight <= 0.0 {
            output.push(None);
            continue;
        }

        output.push(rolling_quantile_from_active(
            &sorted_points,
            &active_weights,
            &weight_tree,
            &count_tree,
            quantile,
            total_weight,
        ));
    }

    output
}

fn active_ordinal_before(count_tree: &FenwickTree, idx: usize) -> f64 {
    if idx == 0 {
        0.0
    } else {
        count_tree.prefix_sum(idx - 1)
    }
}

fn active_by_ordinal(count_tree: &FenwickTree, ordinal_one_based: f64) -> Option<usize> {
    if ordinal_one_based <= 0.0 || ordinal_one_based > count_tree.total() {
        None
    } else {
        count_tree.first_count_ge(ordinal_one_based)
    }
}

fn next_active(count_tree: &FenwickTree, idx: usize) -> Option<usize> {
    active_by_ordinal(count_tree, active_ordinal_before(count_tree, idx) + 2.0)
}

fn prev_active(count_tree: &FenwickTree, idx: usize) -> Option<usize> {
    active_by_ordinal(count_tree, active_ordinal_before(count_tree, idx))
}

fn first_active(count_tree: &FenwickTree) -> Option<usize> {
    active_by_ordinal(count_tree, 1.0)
}

fn last_active(count_tree: &FenwickTree) -> Option<usize> {
    active_by_ordinal(count_tree, count_tree.total())
}

fn midpoint_at(weight_tree: &FenwickTree, active_weights: &[f64], idx: usize) -> f64 {
    weight_tree.prefix_sum(idx) - active_weights[idx] / 2.0
}

fn rolling_quantile_from_active(
    sorted_points: &[RollingPoint],
    active_weights: &[f64],
    weight_tree: &FenwickTree,
    count_tree: &FenwickTree,
    quantile: f64,
    total_weight: f64,
) -> Option<f64> {
    let target = quantile.clamp(0.0, 1.0) * total_weight;

    let mut upper_idx = if target <= 0.0 {
        first_active(count_tree)?
    } else {
        weight_tree
            .first_prefix_ge(target)
            .and_then(|idx| {
                if active_weights[idx] > 0.0 {
                    Some(idx)
                } else {
                    next_active(count_tree, idx).or_else(|| prev_active(count_tree, idx))
                }
            })
            .or_else(|| last_active(count_tree))?
    };

    if midpoint_at(weight_tree, active_weights, upper_idx) < target {
        upper_idx = next_active(count_tree, upper_idx).unwrap_or(upper_idx);
    }

    let upper_midpoint = midpoint_at(weight_tree, active_weights, upper_idx);
    let lower_idx = if upper_midpoint <= target {
        upper_idx
    } else {
        prev_active(count_tree, upper_idx).unwrap_or(upper_idx)
    };

    let lower_midpoint = midpoint_at(weight_tree, active_weights, lower_idx);
    let x_lower = sorted_points[lower_idx].value;
    let x_upper = sorted_points[upper_idx].value;

    if lower_midpoint == upper_midpoint {
        return Some(x_lower);
    }

    let fraction = (target - lower_midpoint) / (upper_midpoint - lower_midpoint);
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

    #[test]
    fn rolling_matches_window_impl() {
        let values = ca(&[
            Some(5.0),
            Some(1.0),
            Some(4.0),
            Some(2.0),
            Some(3.0),
            Some(10.0),
        ]);
        let weights = ca(&[
            Some(1.0),
            Some(2.0),
            Some(1.0),
            Some(3.0),
            Some(1.0),
            Some(1.0),
        ]);

        let result = rolling_weighted_quantile_impl(&values, &weights, 3, 0.5, 3);
        for idx in 0..values.len() {
            let expected = if idx + 1 < 3 {
                None
            } else {
                let start = idx + 1 - 3;
                let window_values = ca(
                    &(start..=idx)
                        .map(|window_idx| values.get(window_idx))
                        .collect::<Vec<_>>(),
                );
                let window_weights = ca(
                    &(start..=idx)
                        .map(|window_idx| weights.get(window_idx))
                        .collect::<Vec<_>>(),
                );
                weighted_quantile_impl(&window_values, &window_weights, 0.5, 3)
            };

            match (result[idx], expected) {
                (Some(left), Some(right)) => assert!((left - right).abs() < 1e-12),
                (None, None) => {}
                other => panic!("mismatch at {idx}: {other:?}"),
            }
        }
    }
}

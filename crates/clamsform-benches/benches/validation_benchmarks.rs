use clamsform_core::validation::errors::{
    validate_dataframe,
    validate_dataframe_parallel,
};
use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
};
use polars::prelude::*;
use rand::rngs::StdRng;
use rand::{
    Rng,
    SeedableRng,
};

fn create_test_dataframe(size: usize) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(42);

    let numeric_col: Vec<f64> = (0..size)
        .map(|_| rng.random_range(-1000.0..1000.0))
        .collect();

    let nan_col: Vec<f64> = (0..size)
        .map(|i| {
            if i % 100 == 0 {
                f64::NAN
            } else {
                rng.random_range(-1000.0..1000.0)
            }
        })
        .collect();

    let inf_col: Vec<f64> = (0..size)
        .map(|i| {
            if i % 200 == 0 {
                f64::INFINITY
            } else if i % 201 == 0 {
                f64::NEG_INFINITY
            } else {
                rng.random_range(-1000.0..1000.0)
            }
        })
        .collect();

    df! {
        "numeric" => numeric_col,
        "with_nans" => nan_col,
        "with_infs" => inf_col,
    }
    .unwrap()
}

fn get_test_dataframes() -> Vec<(String, DataFrame)> {
    vec![
        ("small".to_string(), create_test_dataframe(1_000)),
        ("medium".to_string(), create_test_dataframe(100_000)),
        ("large".to_string(), create_test_dataframe(1_000_000)),
        ("xlarge".to_string(), create_test_dataframe(100_000_000)),
    ]
}

fn validation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("DataFrame Validation");

    for (size_name, df) in get_test_dataframes() {
        group.bench_with_input(BenchmarkId::new("sequential", &size_name), &df, |b, df| {
            b.iter(|| validate_dataframe(df))
        });

        group.bench_with_input(BenchmarkId::new("parallel", &size_name), &df, |b, df| {
            b.iter(|| validate_dataframe_parallel(df))
        });
    }
    group.finish();
}

criterion_group!(benches, validation_benchmark);
criterion_main!(benches);

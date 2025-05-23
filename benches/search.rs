#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use cinder::search::{Control, Depth, Engine, Limits, Options};
use cinder::{nnue::Evaluator, util::Integer};
use criterion::{Criterion, SamplingMode, Throughput};
use criterion_macro::criterion;
use futures::executor::block_on_stream;
use std::thread::available_parallelism;
use std::time::{Duration, Instant};

fn bench(reps: u64, options: &Options, limits: &Limits) -> Duration {
    let position = Evaluator::default();
    let mut time = Duration::ZERO;

    for _ in 0..reps {
        let mut engine = Engine::with_options(options);
        let ctrl = Control::new(&position, limits.clone());
        let search = engine.search(&position, &ctrl);

        let timer = Instant::now();
        block_on_stream(search).for_each(drop);
        time += timer.elapsed();
    }

    time
}

#[criterion]
fn crit(c: &mut Criterion) {
    let thread_limit = match available_parallelism() {
        Ok(cores) => cores.get().div_ceil(2),
        Err(_) => 1,
    };

    let options = Vec::from_iter((0..=thread_limit.ilog2()).map(|threads| Options {
        threads: 2usize.pow(threads).saturate(),
        ..Options::default()
    }));

    for o in &options {
        let depth = Depth::new(18);
        c.benchmark_group("ttd")
            .sampling_mode(SamplingMode::Flat)
            .bench_function(o.threads.to_string(), |b| {
                b.iter_custom(|i| bench(i, o, &depth.into()))
            });
    }

    for o in &options {
        let nodes = 250_000;
        c.benchmark_group("nps")
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(nodes))
            .bench_function(o.threads.to_string(), |b| {
                b.iter_custom(|i| bench(i, o, &nodes.into()))
            });
    }
}

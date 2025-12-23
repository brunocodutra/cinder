#![allow(long_running_const_eval)]
#![feature(exit_status_error)]

use anyhow::{Context, Error as Failure};
use cinder::params::Params;
use cinder::util::{StaticSeq, parsers::*};
use clap::{Args, Parser, Subcommand};
use nom::{Parser as _, bytes::complete::*, character::complete::*, multi::*, sequence::*};
use rand::prelude::*;
use ron::de::from_reader as deserialize;
use ron::ser::{PrettyConfig, to_string_pretty as serialize};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Display, Formatter};
use std::io::{Write, stdout};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use std::{fs::File, num::NonZero, ops::Div, process::Command, thread::available_parallelism};

/// Configuration for the SPSA algorithm.
#[derive(Debug, Args, Serialize, Deserialize)]
struct SpsaConfig {
    /// A ratio coefficient.
    #[clap(long, default_value_t = 0.1)]
    ratio: f32,

    /// Alpha coefficient.
    #[clap(long, default_value_t = 0.602)]
    alpha: f32,

    /// Gamma coefficient.
    #[clap(long, default_value_t = 0.101)]
    gamma: f32,

    /// Final step size as a fraction of parameter range.
    #[clap(long, default_value_t = 0.05)]
    grain: f32,

    /// Final learning rate.
    #[clap(long, default_value_t = 0.002)]
    lr: f32,
}

/// Result of a chess game series.
#[derive(Debug, Default, Copy, Clone)]
struct GameResult {
    wins: u32,
    losses: u32,
}

/// Parse a single key-value pair
fn parse_key_val(s: &str) -> Result<(String, String), Failure> {
    let mid = s.find('=').context("expected `key=val` pair")?;
    Ok((s[..mid].to_owned(), s[mid + 1..].to_owned()))
}

/// Orchestrates a series of game between engines.
#[derive(Debug, Args, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatchRunner {
    /// How many games to run concurrently per match.
    #[clap(long, default_value_t = available_parallelism().map_or(1, NonZero::get).div(2).max(1))]
    concurrency: usize,

    /// The number of game pairs per iteration.
    #[clap(long)]
    pairs: u32,

    /// Path to the opening book file.
    #[clap(long)]
    opening: Option<PathBuf>,

    /// Path to syzygy table base files.
    #[clap(long)]
    syzygy: Option<PathBuf>,

    /// Path to the engine.
    #[clap(long)]
    engine: PathBuf,

    /// Options for the engine.
    #[clap(long, value_parser = parse_key_val, value_delimiter = ',')]
    options: Vec<(String, String)>,

    /// Time control.
    #[clap(long)]
    tc: String,
}

impl MatchRunner {
    fn pairs_per_match(&self) -> u32 {
        self.pairs
    }

    fn run(&self, left: &Params, right: &Params) -> Result<GameResult, Failure> {
        let pairs = self.pairs;
        let concurrency = self.concurrency;
        let engine = self.engine.display();
        let tc = self.tc.as_str();

        let openings = self.opening.as_ref().map_or_else(String::new, |p| {
            format!("-openings file={} order=random", p.display())
        });

        let (tb, mut options) = self.syzygy.as_ref().map_or_else(Default::default, |p| {
            let path = p.display();
            (format!("-tb {path}"), format!("option.SyzygyPath={path}"))
        });

        options.extend(self.options.iter().map(|(k, v)| format!(" option.{k}={v}")));

        let args = format!(
            "-games 2 -rounds {pairs} -concurrency {concurrency} -use-affinity -recover
            -report penta=false -ratinginterval 0 -autosaveinterval 0 {openings} {tb}
            -draw movenumber=32 movecount=6 score=15 -resign movecount=5 score=600
            -engine name=left cmd={engine} args=--params={left}
            -engine name=right cmd={engine} args=--params={right}
            -each tc={tc} {options}"
        );

        let args: Vec<_> = args.split_whitespace().collect();
        let output = Command::new("fastchess").args(&args).output()?;
        let stdout = String::from_utf8(output.stdout)?;

        if let Err(e) = output.status.exit_ok() {
            eprintln!("{stdout}");
            return Err(e).context(String::from_utf8(output.stderr)?);
        }

        let wins = field("Wins:", int);
        let losses = field("Losses:", int);
        let mut parser = many_till(anychar, separated_pair(wins, t(tag(",")), losses));
        let (_, (_, (wins, losses))) = parser.parse(&stdout).map_err(|e| e.to_owned())?;

        Ok(GameResult {
            wins: wins.try_into()?,
            losses: losses.try_into()?,
        })
    }
}

#[derive(Debug)]
struct DurationFormatter(Duration);

impl Display for DurationFormatter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let days = self.0.as_secs() / 86400;
        let hours = (self.0.as_secs() / 3600) % 24;
        let minutes = (self.0.as_secs() / 60) % 60;
        let seconds = self.0.as_secs() % 60;

        if days > 0 {
            write!(f, "{days}d ")?;
        }

        if days > 0 || hours > 0 {
            write!(f, "{hours:02}h ")?;
        }

        if days > 0 || hours > 0 || minutes > 0 {
            write!(f, "{minutes:02}m ")?;
        }

        write!(f, "{seconds:02}s")
    }
}

/// SPSA tuner implementation.
#[derive(Debug, Args, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpsaTuner {
    /// The total number of iterations to run.
    #[clap(long)]
    iters: u32,

    #[clap(flatten)]
    config: SpsaConfig,

    #[clap(flatten)]
    runner: MatchRunner,

    #[clap(skip)]
    params: Params,

    #[clap(skip)]
    step: u32,

    #[clap(skip)]
    period: Duration,
}

impl SpsaTuner {
    fn gradient(&self, left: &Params, right: &Params) -> Result<f32, Failure> {
        let result = self.runner.run(left, right)?;
        Ok(result.losses as f32 - result.wins as f32)
    }

    fn step(&mut self) -> Result<(), Failure> {
        let games_per_step = 2. * self.runner.pairs_per_match() as f32;
        let n = games_per_step * self.iters as f32;
        let k = games_per_step * self.step as f32;
        let a = n * self.config.ratio;

        let a0 = self.config.lr * self.config.grain.powf(2.) * (n + a).powf(self.config.alpha);
        let c0 = self.config.grain * n.powf(self.config.gamma);

        let ak = a0 / (k + a).powf(self.config.alpha);
        let ck = c0 / k.powf(self.config.gamma);

        let mut rng = rand::rng();
        let delta: StaticSeq<_, { Params::LEN }> = (0..Params::LEN)
            .map(|_| if rng.random() { ck } else { -ck })
            .collect();

        let (left, right) = self.params.perturb(delta.iter().copied());

        let gradient = self.gradient(&left, &right)?;
        let correction = delta.into_iter().map(|d| -ak * gradient / d);
        self.params.update(correction);

        Ok(())
    }

    fn run<P: AsRef<Path>>(&mut self, filename: P) -> Result<(), Failure> {
        let content = serialize(self, PrettyConfig::default().compact_arrays(true))?;
        let mut file = File::create(&filename)?;
        write!(file, "{content}")?;

        let mut stdout = stdout().lock();
        while self.step < self.iters {
            self.step += 1;
            let timer = Instant::now();
            self.step()?;
            let elapsed = timer.elapsed();
            self.period = (self.period.saturating_mul(self.step - 1) + elapsed) / self.step;

            let content = serialize(self, PrettyConfig::default().compact_arrays(true))?;
            let mut file = File::create(&filename)?;
            write!(file, "{content}")?;

            let period = DurationFormatter(self.period);
            let remaining = DurationFormatter(self.period.saturating_mul(self.iters - self.step));

            write!(stdout, "steps completed: {}/{}, ", self.step, self.iters)?;
            write!(stdout, "average time per step: {period}, ")?;
            writeln!(stdout, "estimated time remaining: {remaining}")?;
        }

        let params = serialize(&self.params, PrettyConfig::default().compact_arrays(true))?;
        writeln!(stdout, "optimized parameters: {params}")?;

        Ok(())
    }
}

/// A Simultaneous Perturbation Stochastic Approximation hyper-parameter tuner.
#[derive(Debug, Parser)]
struct Cli {
    /// The checkpoint file.
    #[clap(long, default_value = "spsa.ron")]
    checkpoint: PathBuf,

    #[clap(subcommand)]
    action: Action,
}

/// Controls whether to resume a tuning session from a checkpoint or to start a new one.
#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Action {
    /// Starts a new tuning session.
    Start(#[clap(flatten)] SpsaTuner),

    /// Resumes from the checkpoint.
    Resume,
}

fn main() -> Result<(), Failure> {
    let args = Cli::parse();
    let mut spsa = match args.action {
        Action::Start(spsa) => spsa,
        Action::Resume => {
            let file = File::open(&args.checkpoint)?;
            deserialize(file)?
        }
    };

    spsa.run(args.checkpoint)
}

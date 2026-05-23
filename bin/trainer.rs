#![allow(long_running_const_eval)]

use anyhow::{Context, Error as Failure};
use bullet::game::formats::bulletformat::ChessBoard;
use bullet::game::formats::sfbinpack::TrainingDataEntry;
use bullet::game::formats::sfbinpack::chess::r#move::MoveType;
use bullet::game::formats::sfbinpack::chess::piecetype::PieceType;
use bullet::game::{inputs::SparseInputType, outputs::OutputBuckets};
use bullet::lr::{LinearDecayLR, Warmup};
use bullet::nn::optimiser::{AdamW, AdamWParams};
use bullet::nn::{InitSettings, Shape};
use bullet::trainer::schedule::{TrainingSchedule, TrainingSteps};
use bullet::trainer::{save::SavedFormat, settings::LocalSettings};
use bullet::value::ValueTrainerBuilder;
use bullet::value::loader::SfBinpackLoader;
use bullet::wdl::LinearWDL;
use bullet_trainer::reader::DataReader;
use bytemuck::zeroed;
use cinder::chess::{Color, Flip, Phase, Piece, Role, Square};
use cinder::{nnue::*, util::Num};
use clap::{Args, Parser, Subcommand};
use rand::{prelude::*, rng};
use std::ops::{Deref, Div, RangeInclusive};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{cell::Cell, fs::create_dir_all, num::NonZero, thread::available_parallelism};

const fn spline(p: f64, points: &[(f64, f64)]) -> f64 {
    let mut i = 0;
    while i < points.len() - 1 {
        if p >= points[i].0 && p < points[i + 1].0 {
            let t = (p - points[i].0) / (points[i + 1].0 - points[i].0);
            return points[i].1 + t * (points[i + 1].1 - points[i].1);
        }

        i += 1;
    }

    if p < points[0].0 {
        points[0].1
    } else if p >= points[points.len() - 1].0 {
        points[points.len() - 1].1
    } else {
        panic!()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct KingBuckets;

impl KingBuckets {
    const LEN: usize = Bucket::LEN / 2;
}

impl SparseInputType for KingBuckets {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        Feature::LEN
    }

    fn max_active(&self) -> usize {
        32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let our_king = <Square as Num>::new(pos.our_ksq() as _);
        let opp_king = <Square as Num>::new(pos.opp_ksq() as _).flip();

        for (p, sq) in pos.into_iter() {
            let sq = <Square as Num>::new(sq as _);
            let piece = Piece::new(Role::new(p & 7), Color::new((p & 8) >> 3));

            f(
                Feature::new(Color::White, our_king, piece, sq).cast(),
                Feature::new(Color::Black, opp_king, piece, sq).cast(),
            )
        }
    }

    fn shorthand(&self) -> String {
        format!("768x{}hm", Self::LEN)
    }

    fn description(&self) -> String {
        "Horizontally mirrored, king bucketed psqt chess inputs".to_owned()
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Phaser;

impl OutputBuckets<ChessBoard> for Phaser {
    const BUCKETS: usize = Phase::LEN;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        (pos.occ().count_ones() as u8 - 1) / 4
    }
}

/// The configuration for a filter that can be applied to a game during unpacking.
#[derive(Debug, Clone, Args)]
pub struct TrainingDataFilter {
    /// Filter out positions that have an absolute score above this value.
    #[clap(long, default_value_t = 10000)]
    pub max_score: u16,
    /// Filter out positions where score diverges from the result by more than this value.
    #[clap(long, default_value_t = 2500)]
    pub max_score_anomaly: u16,
    /// The probability of skipping a random position.
    #[clap(long, default_value_t = 0.25)]
    pub random_rejection: f64,
    /// Whether to enable adaptive piece count filtering.
    #[clap(long, default_value_t = true)]
    pub piece_count_filter: bool,
    /// Whether to skip positions based on the WDL model.
    #[clap(long, default_value_t = true)]
    pub wdl_filter: bool,
    /// Whether to skip positions based on the ply.
    #[clap(long, default_value_t = true)]
    pub ply_filter: bool,
}

impl TrainingDataFilter {
    /// How likely the end-game result predicted from the current score is.
    fn predicted_result_chance(&self, entry: &TrainingDataEntry) -> f64 {
        /// This win rate model returns the probability of winning given the score
        /// and a game-ply. The model fits rather accurately the LTC fishtest statistics.
        const WDL_MODEL_PARAMS_A: [f64; 4] = [-3.68389304, 30.07065921, -60.52878723, 149.53378557];
        const WDL_MODEL_PARAMS_B: [f64; 4] = [-2.01818570, 15.85685038, -29.83452023, 47.59078827];
        const WDL_MODEL_PARAMS_B_SCALE: f64 = 1.5;
        const WDL_MODEL_MAX_PLY: u16 = 240;
        const WDL_MODEL_PLY_SCALE: f64 = 64.;
        const WDL_MODEL_MAX_SCORE: f64 = 2000.;
        const WDL_MODEL_SCORE_SCALE: f64 = 0.4807692307692308;

        let m = entry.ply.min(WDL_MODEL_MAX_PLY) as f64 / WDL_MODEL_PLY_SCALE;

        let a = WDL_MODEL_PARAMS_A[0]
            .mul_add(m, WDL_MODEL_PARAMS_A[1])
            .mul_add(m, WDL_MODEL_PARAMS_A[2])
            .mul_add(m, WDL_MODEL_PARAMS_A[3]);

        let b = WDL_MODEL_PARAMS_B[0]
            .mul_add(m, WDL_MODEL_PARAMS_B[1])
            .mul_add(m, WDL_MODEL_PARAMS_B[2])
            .mul_add(m, WDL_MODEL_PARAMS_B[3]);

        let b = b * WDL_MODEL_PARAMS_B_SCALE;

        let x = f64::clamp(
            entry.score as f64 * WDL_MODEL_SCORE_SCALE,
            -WDL_MODEL_MAX_SCORE,
            WDL_MODEL_MAX_SCORE,
        );

        let w = 1.0 / (1.0 + f64::exp((a - x) / b));
        let l = 1.0 / (1.0 + f64::exp((a + x) / b));
        let d = 1.0 - w - l;

        match entry.result {
            1.. => w,
            0 => d,
            ..0 => l,
        }
    }

    /// Probability of rejecting a position based on wdl deviation
    fn predicted_result_rejection(&self, entry: &TrainingDataEntry) -> f64 {
        1.0 - self.predicted_result_chance(entry).clamp(0.0, 1.0)
    }

    /// Adaptive piece count filtering to maintain desired distribution.
    fn piece_count_rejection(&self, entry: &TrainingDataEntry) -> f64 {
        #[rustfmt::skip]
        const DESIRED_DISTRIBUTION: [f64; 33] = [
            0.018411966423, 0.020641545085, 0.022727271053,
            0.024669162740, 0.026467201733, 0.028121406444,
            0.029631758462, 0.030998276198, 0.032220941240,
            0.033299772000, 0.034234750067, 0.035025893853,
            0.035673184944, 0.036176641754, 0.036536245870,
            0.036752015705, 0.036823932846, 0.036752015705,
            0.036536245870, 0.036176641754, 0.035673184944,
            0.035025893853, 0.034234750067, 0.033299772000,
            0.032220941240, 0.030998276198, 0.029631758462,
            0.028121406444, 0.026467201733, 0.024669162740,
            0.022727271053, 0.020641545085, 0.018411966423,
        ];

        static PIECE_COUNT_STATS: [AtomicU64; 33] = zeroed();
        static PIECE_COUNT_TOTAL: AtomicU64 = AtomicU64::new(0);

        let pc = entry.pos.occupied().count() as usize;
        let count = PIECE_COUNT_STATS[pc].fetch_add(1, Ordering::Relaxed) + 1;
        let total = PIECE_COUNT_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        let frequency = count as f64 / total as f64;

        // Calculate the acceptance probability for this piece count
        let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
        1.0 - acceptance.clamp(0.0, 1.0)
    }

    /// Whether we consider this ply too early.
    fn early_ply_rejection(&self, entry: &TrainingDataEntry) -> f64 {
        const EARLY_PLY_ACCEPTANCE: [f64; 31] = {
            let mut table = [0.0f64; 31];

            let points = [(12.0, 0.0), (16.0, 0.4), (18.0, 0.65), (20.0, 1.0)];

            let mut i = 0;
            while i < table.len() {
                table[i] = spline(i as f64, &points).clamp(0.0, 1.0);
                i += 1;
            }

            table
        };

        let ply = entry.ply as usize;
        if ply < EARLY_PLY_ACCEPTANCE.len() {
            1.0 - EARLY_PLY_ACCEPTANCE[ply]
        } else {
            0.0
        }
    }

    /// By how much this position's score deviates from the game result.
    fn score_anomaly(&self, entry: &TrainingDataEntry) -> i16 {
        match entry.result {
            0 => entry.score,
            r => i16::min(r.signum() * entry.score, 0),
        }
    }

    /// Whether the position score doesn't seem trustworthy.
    fn is_suspicious_score(&self, entry: &TrainingDataEntry) -> bool {
        thread_local! {
            static LAST_PLY: Cell<i32> = const { Cell::new(-1) };
            static LAST_SCORE: Cell<Option<i16>> = const { Cell::new(None) };
        }

        // Detect placeholder zero: a position where score=0 was written
        // because the entry is to be skipped, not a genuine eval.
        let is_placeholder_zero = entry.ply as i32 > LAST_PLY.get()
            && LAST_SCORE.get().is_some_and(|s| s.abs() > 100)
            && entry.result != 0
            && entry.score == 0;

        LAST_PLY.set(entry.ply as i32);

        if is_placeholder_zero {
            return true;
        }

        LAST_SCORE.set(Some(entry.score));

        entry.score.unsigned_abs() > self.max_score
            || self.score_anomaly(entry).unsigned_abs() > self.max_score_anomaly
    }

    /// Whether this position is tactical or forced.
    fn is_noisy_position(&self, entry: &TrainingDataEntry) -> bool {
        entry.pos.is_checked(entry.pos.side_to_move())
            || entry.mv.mtype() != MoveType::Normal
            || entry.pos.piece_at(entry.mv.to()).piece_type() != PieceType::None
    }

    fn should_skip(&self, entry: &TrainingDataEntry) -> bool {
        let mut rng = rng();

        // IMPORTANT: evaluated unconditionally due to thread-local side-effect.
        let is_suspicious_score = self.is_suspicious_score(entry);

        is_suspicious_score
            || self.is_noisy_position(entry)
            || rng.random_bool(self.random_rejection)
            || (self.ply_filter && rng.random_bool(self.early_ply_rejection(entry)))
            || (self.wdl_filter && rng.random_bool(self.predicted_result_rejection(entry)))
            || (self.piece_count_filter && rng.random_bool(self.piece_count_rejection(entry)))
    }
}

const SB0: usize = 200;
const SB1: usize = 600;
const SB2: usize = 200;

/// An efficiently updatable neural network (NNUE) trainer.
#[derive(Debug, Parser)]
struct Orchestrator {
    /// The path where to store checkpoints.
    #[clap(long, default_value = "checkpoints/")]
    checkpoints: String,

    /// How many threads to use for data loading.
    #[clap(long, default_value_t = available_parallelism().map_or(1, NonZero::get).div(4).max(1))]
    threads: usize,

    /// How many positions per batch.
    #[clap(long, default_value_t = 131072)]
    batch_size: usize,

    /// How many batches per superbatch.
    #[clap(long, default_value_t = 768)]
    batches_per_superbatch: usize,

    /// The target wdl fraction.
    #[clap(long, default_value_t = 0.25)]
    wdl: f32,

    #[clap(flatten)]
    filter: TrainingDataFilter,

    /// The datasets to use for training.
    #[clap(long, value_delimiter = ',')]
    datasets: Vec<String>,

    /// Whether to start from scratch or resume from a checkpoint.
    #[clap(subcommand)]
    mode: Mode,
}

/// Controls whether to resume training from a checkpoint.
#[derive(Debug, Subcommand)]
enum Mode {
    /// Start training from scratch.
    Start,

    /// Resumes from a checkpoint.
    Resume {
        /// The stage to resume from
        #[clap(long)]
        stage: usize,
        /// The superbatch to resume from
        #[clap(long)]
        superbatch: usize,
    },
}

impl Orchestrator {
    fn dataloader(&self, data: &[&str]) -> impl DataReader<ChessBoard> {
        let filter = self.filter.clone();
        SfBinpackLoader::new_concat_multiple(data, 1024, self.threads, move |entry| {
            !filter.should_skip(entry)
        })
    }

    fn schedule(
        &self,
        id: &str,
        sb: RangeInclusive<usize>,
        lr: RangeInclusive<f32>,
        wdl: RangeInclusive<f32>,
    ) -> TrainingSchedule<Warmup<LinearDecayLR>, LinearWDL> {
        let (start_superbatch, end_superbatch) = sb.into_inner();
        let (initial_lr, final_lr) = lr.into_inner();
        let (start_wdl, end_wdl) = wdl.into_inner();

        TrainingSchedule {
            net_id: id.to_string(),
            eval_scale: 1.0,
            steps: TrainingSteps {
                batch_size: self.batch_size,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch,
                end_superbatch,
            },
            wdl_scheduler: LinearWDL {
                start: start_wdl,
                end: end_wdl,
            },
            lr_scheduler: Warmup {
                warmup_batches: self.batches_per_superbatch,
                inner: LinearDecayLR {
                    initial_lr,
                    final_lr,
                    final_superbatch: end_superbatch,
                },
            },
            save_rate: 10,
        }
    }

    fn run(&self) -> Result<(), Failure> {
        let mut trainer = ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs(KingBuckets)
            .output_buckets(Phaser)
            .save_format(&[
                SavedFormat::id("ftw")
                    .transform(|store, weights| {
                        let factoriser = store.get("ftf").values.f32().repeat(KingBuckets::LEN);
                        Vec::from_iter(weights.into_iter().zip(factoriser).map(|(w, f)| w + f))
                    })
                    .round()
                    .quantise::<i16>(FTQ),
                SavedFormat::id("ftb").round().quantise::<i16>(FTQ),
                SavedFormat::id("l12w")
                    .transpose()
                    .round()
                    .quantise::<i8>(HLS),
                SavedFormat::id("l12b"),
                SavedFormat::id("r2o").transpose(),
                SavedFormat::id("l23w").transpose(),
                SavedFormat::id("l23b"),
                SavedFormat::id("l34w").transpose(),
                SavedFormat::id("l34b"),
                SavedFormat::id("l4ow").transpose(),
                SavedFormat::id("l4ob"),
            ])
            .use_win_rate_model()
            .build_custom(|builder, (stm, ntm, phase), target| {
                let shape = Shape::new(Accumulator::LEN, Feature::LEN / KingBuckets::LEN);
                let ftf = builder.new_weights("ftf", shape, InitSettings::Zeroed);

                let mut ft = builder.new_affine("ft", Feature::LEN, Accumulator::LEN);
                ft.init_with_effective_input_size(32);

                let max_weight = i16::MAX as f32 / 32. / FTQ as f32;
                ft.weights = (ft.weights + ftf.repeat(KingBuckets::LEN))
                    .clip_pass_through_grad(-max_weight, max_weight);

                let l12 = builder.new_affine("l12", Li::LEN, Phase::LEN * Ln::LEN / 2);
                let l23 = builder.new_affine("l23", Ln::LEN, Phase::LEN * Ln::LEN / 2);
                let l34 = builder.new_affine("l34", Ln::LEN, Phase::LEN * Ln::LEN / 2);
                let l4o = builder.new_affine("l4o", Ln::LEN, Phase::LEN);

                let shape = Shape::new(Phase::LEN, Ln::LEN);
                let r2o = builder.new_weights("r2o", shape, InitSettings::Zeroed);

                let ft = |input, start, end| ft.slice(start, end).forward(input).crelu();
                let stm = ft(stm, 0, Li::LEN / 2) * ft(stm, Li::LEN / 2, Li::LEN);
                let ntm = ft(ntm, 0, Li::LEN / 2) * ft(ntm, Li::LEN / 2, Li::LEN);

                let l1a = stm.concat(ntm);
                let l2 = l12.forward(l1a).select(phase);
                let l2a = l2.concat(-l2).sqrrelu();
                let l3 = l23.forward(l2a).select(phase);
                let l3a = l3.concat(-l3).sqrrelu();
                let l4 = l34.forward(l3a).select(phase);
                let l4a = l4.concat(-l4).sqrrelu();
                let out = l4o.forward(l4a).select(phase) + r2o.matmul(l2a).select(phase);

                let ones = builder.new_constant(Shape::new(1, Li::LEN), &[1.0; Li::LEN]);
                let l1_reg = ones.matmul(l1a) / Li::LEN as f32;

                let score = 300.0 * out;
                let qp = (score - 270.0) / 340.0;
                let qn = (-score - 270.0) / 340.0;
                let inferred = 0.5 * (1.0 + qp.sigmoid() - qn.sigmoid());

                let err = inferred - target;
                let err_relu = err.relu();
                (out, err * err + 0.15 * err_relu * err_relu + 0.005 * l1_reg)
            });

        let params = AdamWParams {
            min_weight: f32::MIN,
            max_weight: f32::MAX,
            ..AdamWParams::default()
        };

        let max_weight = HLQ as f32 / HLS as f32;
        let clipped = AdamWParams {
            min_weight: -max_weight,
            max_weight,
            ..params
        };

        trainer.optimiser.set_params(params);
        trainer.optimiser.set_params_for_weight("l12w", clipped);

        let settings = LocalSettings {
            threads: self.threads,
            test_set: None,
            output_directory: &self.checkpoints,
            batch_queue_size: self.batches_per_superbatch,
        };

        let (stage, superbatch) = match self.mode {
            Mode::Resume { stage, superbatch } => (stage, superbatch),
            Mode::Start => (0, 0),
        };

        create_dir_all(&self.checkpoints)?;
        if matches!(self.mode, Mode::Resume { .. }) {
            let checkpoint = format!("{}/stage{}-{}", self.checkpoints, stage, superbatch);
            trainer.load_from_checkpoint(&checkpoint);
        }

        let priming_dataset = self.datasets.first().context("expected dataset")?;
        let training_datasets = Vec::from_iter(self.datasets[1..].iter().map(Deref::deref));
        let training_dataloader = self.dataloader(&training_datasets);

        if stage == 0 && superbatch < SB0 {
            let start = if stage == 0 { superbatch + 1 } else { 1 };
            let schedule = self.schedule("stage0", start..=SB0, 2e-3..=1e-4, 0.0..=0.0);
            let priming_dataloader = self.dataloader(&[priming_dataset]);
            trainer.run(&schedule, &settings, &priming_dataloader);
        }

        if stage < 1 || (stage == 1 && superbatch < SB1) {
            let start = if stage == 1 { superbatch + 1 } else { 1 };
            let schedule = self.schedule("stage1", start..=SB1, 5e-4..=1e-5, 0.0..=self.wdl);
            trainer.run(&schedule, &settings, &training_dataloader);
        }

        if stage < 2 || (stage == 2 && superbatch < SB2) {
            let start = if stage == 2 { superbatch + 1 } else { 1 };
            let schedule = self.schedule("stage2", start..=SB2, 2e-5..=1e-6, self.wdl..=self.wdl);
            trainer.run(&schedule, &settings, &training_dataloader);
        }

        Ok(())
    }
}

fn main() -> Result<(), Failure> {
    Orchestrator::parse().run()
}

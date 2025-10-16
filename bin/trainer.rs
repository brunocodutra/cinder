use anyhow::{Context, Error as Failure};
use bullet::game::formats::bulletformat::ChessBoard;
use bullet::game::formats::sfbinpack::TrainingDataEntry;
use bullet::game::formats::sfbinpack::chess::r#move::MoveType;
use bullet::game::formats::sfbinpack::chess::piecetype::PieceType;
use bullet::game::inputs::SparseInputType;
use bullet::game::outputs::OutputBuckets;
use bullet::lr::{LinearDecayLR, Warmup};
use bullet::nn::optimiser::{AdamW, AdamWParams};
use bullet::nn::{InitSettings, Shape};
use bullet::trainer::save::SavedFormat;
use bullet::trainer::schedule::{TrainingSchedule, TrainingSteps};
use bullet::trainer::settings::LocalSettings;
use bullet::value::ValueTrainerBuilder;
use bullet::value::loader::{DataLoader, SfBinpackLoader};
use bullet::wdl::LinearWDL;
use bytemuck::zeroed;
use cinder::chess::{Color, Flip, Phase, Piece, Role, Square};
use cinder::{nnue::*, util::Integer};
use clap::{Args, Parser, Subcommand};
use rand::{Rng, rng};
use std::ops::{Deref, Div, RangeInclusive};
use std::sync::atomic::{AtomicU64, Ordering};
use std::{fs::create_dir_all, num::NonZero, thread::available_parallelism};

#[derive(Debug, Default, Copy, Clone)]
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
        let our_king = <Square as Integer>::new(pos.our_ksq() as _);
        let opp_king = <Square as Integer>::new(pos.opp_ksq() as _).flip();

        for (p, sq) in pos.into_iter() {
            let sq = <Square as Integer>::new(sq as _);
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
        "Horizontally mirrored, king bucketed psqt chess inputs".to_string()
    }
}

#[derive(Debug, Default, Copy, Clone)]
struct Phaser;

impl OutputBuckets<ChessBoard> for Phaser {
    const BUCKETS: usize = Phase::LEN;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        (pos.occ().count_ones() as u8 - 1) / 4
    }
}

/// The configuration for a filter that can be applied to a game during unpacking.
#[derive(Debug, Copy, Clone, Args)]
pub struct TrainingDataFilter {
    /// Filter out positions that have a ply count less than this value.
    #[clap(skip = 0u16)]
    pub min_ply: u16,
    /// Filter out positions that have an absolute score above this value.
    #[clap(long, default_value_t = 10000)]
    pub max_score: u16,
    /// Filter out positions where score diverges from the result by more than this value.
    #[clap(long, default_value_t = 1000)]
    pub max_score_anomaly: u16,
    /// The probability of skipping a random position.
    #[clap(long, default_value_t = 0.25)]
    pub random_skip_chance: f64,
    /// Whether to enable adaptive piece count filtering.
    #[clap(long, default_value_t = true)]
    pub piece_count_filter: bool,
    /// Whether to skip positions based on the WDL model.
    #[clap(long, default_value_t = true)]
    pub wdl_filter: bool,
}

impl TrainingDataFilter {
    /// By how much this position's score deviates from the game result.
    fn score_anomaly(entry: &TrainingDataEntry) -> i16 {
        match entry.result {
            0 => entry.score,
            r => i16::min(r.signum() * entry.score, 0),
        }
    }

    /// How likely the end-game result predicted from the current score is.
    fn predicted_result_chance(entry: &TrainingDataEntry) -> f64 {
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

        let w = 1. / (1. + f64::exp((a - x) / b));
        let l = 1. / (1. + f64::exp((a + x) / b));
        let d = (1. - w - l).max(0.);

        match entry.result {
            1.. => w,
            0 => d,
            ..0 => l,
        }
    }

    /// Adaptive piece count filtering to maintain desired distribution.
    fn piece_count_rejection(entry: &TrainingDataEntry) -> f64 {
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
        1. - acceptance.clamp(0., 1.)
    }

    #[allow(clippy::if_same_then_else)]
    pub fn should_skip(&self, entry: &TrainingDataEntry) -> bool {
        let mut rng = rng();

        entry.ply < self.min_ply
            || entry.score.unsigned_abs() > self.max_score
            || Self::score_anomaly(entry).unsigned_abs() > self.max_score_anomaly
            || entry.mv.mtype() != MoveType::Normal
            || entry.pos.piece_at(entry.mv.to()).piece_type() != PieceType::None
            || entry.pos.is_checked(entry.pos.side_to_move())
            || rng.random_bool(self.random_skip_chance)
            || (self.wdl_filter && rng.random_bool(1. - Self::predicted_result_chance(entry)))
            || (self.piece_count_filter && rng.random_bool(Self::piece_count_rejection(entry)))
    }
}

const SB0: usize = 200;
const SB1: usize = 600;
const SB2: usize = 200;
const MAX_WEIGHT: f32 = HLQ as f32 / HLS as f32;

/// An efficiently updatable neural network (NNUE) trainer.
#[derive(Debug, Parser)]
struct Orchestrator {
    /// The path where to store checkpoints.
    #[clap(long, default_value = "checkpoints/")]
    checkpoints: String,

    /// How many threads to use for data loading.
    #[clap(long, default_value_t = available_parallelism().map_or(1, NonZero::get).div(2).max(1))]
    threads: usize,

    /// How many positions per batch.
    #[clap(long, default_value_t = 262144)]
    batch_size: usize,

    /// How many batches per superbatch.
    #[clap(long, default_value_t = 384)]
    batches_per_superbatch: usize,

    /// The target wdl fraction.
    #[clap(long, default_value_t = 0.25)]
    wdl: f32,

    #[clap(flatten)]
    filter: TrainingDataFilter,

    /// The datasets to use for training.
    #[clap(long, value_delimiter = ',')]
    datasets: Vec<String>,

    /// Whether to resume training from a checkpoint.
    #[clap(subcommand)]
    mode: Option<Mode>,
}

/// Controls whether to resume training from a checkpoint.
#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Mode {
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
    fn dataloader(&self, data: &[&str], min_ply: u16) -> impl DataLoader<ChessBoard> {
        let filter = TrainingDataFilter {
            min_ply,
            ..self.filter
        };

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
            eval_scale: 1.,
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
                        let factoriser = store.get("ftf").values.repeat(KingBuckets::LEN);
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
                SavedFormat::id("l23w").transpose(),
                SavedFormat::id("l23b"),
                SavedFormat::id("l3ow").transpose(),
                SavedFormat::id("l3ob"),
            ])
            .use_win_rate_model()
            .loss_fn(|output, pt| {
                let score = 300. * output.copy();
                let q = (score - 270.) / 340.;
                let qm = (-score - 270.) / 340.;
                let qf = 0.5 * (1. + q.sigmoid() - qm.sigmoid());
                qf.squared_error(pt)
            })
            .build(|builder, stm, ntm, phase| {
                let ftf_shape = Shape::new(Layer0::LEN, Feature::LEN / KingBuckets::LEN);
                let ftf = builder.new_weights("ftf", ftf_shape, InitSettings::Zeroed);

                let mut ft = builder.new_affine("ft", Feature::LEN, Layer0::LEN);
                ft.init_with_effective_input_size(32);
                ft.weights = ft.weights + ftf.repeat(KingBuckets::LEN);

                let l12 = builder.new_affine("l12", Layer1::LEN, Phase::LEN * Layer2::LEN);
                let l23 = builder.new_affine("l23", Layer2::LEN * 2, Phase::LEN * Layer3::LEN);
                let l3o = builder.new_affine("l3o", Layer3::LEN, Phase::LEN);

                let stm = ft.forward(stm).crelu().pairwise_mul();
                let ntm = ft.forward(ntm).crelu().pairwise_mul();

                let l1 = stm.concat(ntm);
                let l2 = l12.forward(l1).select(phase);
                let l2 = l2.abs_pow(2.).concat(l2).crelu();
                let l3 = l23.forward(l2).select(phase).screlu();
                l3o.forward(l3).select(phase)
            });

        let params = AdamWParams {
            min_weight: f32::MIN,
            max_weight: f32::MAX,
            ..AdamWParams::default()
        };

        let clipped = AdamWParams {
            min_weight: -MAX_WEIGHT,
            max_weight: MAX_WEIGHT,
            ..params
        };

        trainer.optimiser.set_params(params);
        trainer.optimiser.set_params_for_weight("ftw", clipped);
        trainer.optimiser.set_params_for_weight("l12w", clipped);

        let settings = LocalSettings {
            threads: self.threads,
            test_set: None,
            output_directory: &self.checkpoints,
            batch_queue_size: 64,
        };

        let (stage, superbatch) = match self.mode {
            Some(Mode::Resume { stage, superbatch }) => (stage, superbatch),
            None => (0, 0),
        };

        create_dir_all(&self.checkpoints)?;
        if matches!(self.mode, Some(Mode::Resume { .. })) {
            let checkpoint = format!("{}/stage{}-{}", self.checkpoints, stage, superbatch);
            trainer.load_from_checkpoint(&checkpoint);
        }

        let priming_dataset = self.datasets.first().context("expected dataset")?;
        let training_datasets = Vec::from_iter(self.datasets[1..].iter().map(Deref::deref));
        let training_dataloader = self.dataloader(&training_datasets, 16);

        if stage == 0 && superbatch < SB0 {
            let start = if stage == 0 { superbatch + 1 } else { 1 };
            let schedule = self.schedule("stage0", start..=SB0, 2.5e-3..=1e-4, 0.0..=0.0);
            let priming_dataloader = self.dataloader(&[priming_dataset], 0);
            trainer.run(&schedule, &settings, &priming_dataloader);
        }

        if stage < 1 || (stage == 1 && superbatch < SB1) {
            let start = if stage == 1 { superbatch + 1 } else { 1 };
            let schedule = self.schedule("stage1", start..=SB1, 1e-3..=2e-5, 0.0..=self.wdl);
            trainer.run(&schedule, &settings, &training_dataloader);
        }

        if stage < 2 || (stage == 2 && superbatch < SB2) {
            let start = if stage == 2 { superbatch + 1 } else { 1 };
            let schedule = self.schedule("stage2", start..=SB2, 2e-5..=2e-7, self.wdl..=self.wdl);
            trainer.run(&schedule, &settings, &training_dataloader);
        }

        Ok(())
    }
}

fn main() -> Result<(), Failure> {
    Orchestrator::parse().run()
}

#![allow(long_running_const_eval)]
#![feature(portable_simd)]

use anyhow::{Error as Failure, bail};
use bullet::game::formats::bulletformat::ChessBoard;
use bullet::game::formats::sfbinpack::TrainingDataEntry;
use bullet::game::formats::sfbinpack::chess::{r#move::MoveType, piecetype::PieceType};
use bullet::game::outputs::OutputBuckets;
use bullet::nn::{ExecutionContext, InitSettings, Shape};
use bullet::trainer::schedule::lr::{LinearDecayLR, LrScheduler, Sequence};
use bullet::trainer::schedule::wdl::{ConstantWDL, LinearWDL, WdlScheduler};
use bullet::value::loader::SfBinpackLoader;
use bullet::value::save::{save_to_checkpoint, write_losses};
use bullet_trainer::model::*;
use bullet_trainer::optimiser::Optimiser;
use bullet_trainer::optimiser::adam::{AdamW, AdamWParams};
use bullet_trainer::reader::{DataReader, ReadMapLoader};
use bullet_trainer::run::{DefaultDevice, Step, TrainingSchedule, TrainingSteps, train};
use bytemuck::zeroed;
use cinder::chess::{Bitboard, Board, Color, Phase, Piece, Position, Role, Square};
use cinder::nnue::*;
use cinder::util::{Assume, Int, Num, StaticSeq};
use clap::{Args, Parser, Subcommand};
use derive_more::with_trait::Deref;
use rand::{prelude::*, rng};
use std::cell::{Cell, RefCell};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::ops::{BitAnd, Div, RangeInclusive};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::{fmt::Debug, fs::create_dir_all, num::NonZero, slice, thread::available_parallelism};

trait OrBail<T> {
    fn or_bail(self) -> Result<T, Failure>;
}

impl<T, E: Debug> OrBail<T> for Result<T, E> {
    fn or_bail(self) -> Result<T, Failure> {
        self.map_err(|e| Failure::msg(format!("{e:?}")))
    }
}

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
        let count = PIECE_COUNT_STATS[pc].fetch_add(1, Relaxed) + 1;
        let total = PIECE_COUNT_TOTAL.fetch_add(1, Relaxed) + 1;
        let frequency = count as f64 / total as f64;

        // Calculate the acceptance probability for this piece count
        let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
        1.0 - acceptance.clamp(0.0, 1.0)
    }

    /// Whether we consider this ply too early.
    fn early_ply_rejection(&self, entry: &TrainingDataEntry) -> f64 {
        const EARLY_PLY_ACCEPTANCE: [f64; 31] = const {
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

const SB0: usize = 100;
const SB1: usize = 800;
const SB2: usize = 100;

const FT_CLIP: f32 = i8::MAX as f32 / FTQ as f32;
const HL_CLIP: f32 = i8::MAX as f32 / HLQ as f32;

const TARGET_SCORE_OFFSET: f32 = 289.0;
const TARGET_SCORE_SCALING: f32 = 380.0;
const INFERRED_SCORE_OFFSET: f32 = 0.0;
const INFERRED_SCORE_SCALING: f32 = 97.0;

#[derive(Debug, Default, Clone, Copy)]
struct Phaser;

impl OutputBuckets<ChessBoard> for Phaser {
    const BUCKETS: usize = Phase::LEN;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        (pos.occ().count_ones() as u8 - 1) / 4
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct Features;

impl Features {
    const BUCKETS: usize = KingBucket::LEN / 2;
    const MAX_ACTIVE: usize = PPFeature::MAX_ACTIVE + TIFeature::MAX_ACTIVE + KAFeature::MAX_ACTIVE;
}

impl Features {
    pub fn map_features(
        &self,
        pos: ChessBoard,
        mut pp: impl FnMut(usize, usize),
        mut ti: impl FnMut(usize, usize),
        mut ka: impl FnMut(usize, usize),
    ) {
        use Color::*;
        let pos = Position::from(pos);
        let ksqs = [pos.king(White), pos.king(Black)];
        let pfts = [White, Black].map(|side| PFeature::lut(side, ksqs[side], &pos).to_array());

        let mut remaining = Bitboard::from(pos.by_role(Role::Pawn));
        for s in remaining.iter() {
            remaining &= !s.bitboard();
            for t in PPFeature::WINDOW[s.file()].bitand(remaining).iter() {
                let pft1 = pfts.map(|ft| Num::new(ft[s]));
                let pft2 = pfts.map(|ft| Num::new(ft[t]));

                pp(
                    PPFeature::new(pft1[White], pft2[White]).cast(),
                    PPFeature::new(pft1[Black], pft2[Black]).cast(),
                );
            }
        }

        let occupied = pos.occupied();
        let attacks = pos.threats().map(|t| t.mask(occupied));

        let mut stm_ti_features = StaticSeq::<TIFeature, { TIFeature::MAX_ACTIVE }>::new();
        let mut ntm_ti_features = StaticSeq::<TIFeature, { TIFeature::MAX_ACTIVE }>::new();

        for c in Color::iter() {
            for sq in Square::iter() {
                let indices = attacks[c][sq];
                let dst = pos[sq].piece().assume();

                for idx in indices {
                    let wc = pos.squares()[c][idx].assume();
                    let src = Piece::new(pos.roles()[c][idx].assume(), c);

                    if let Some(f) = TIFeature::new(White, ksqs[White], src, wc, dst, sq) {
                        stm_ti_features.push(f);
                    }

                    if let Some(f) = TIFeature::new(Black, ksqs[Black], src, wc, dst, sq) {
                        ntm_ti_features.push(f);
                    }
                }
            }
        }

        assert_eq!(stm_ti_features.len(), ntm_ti_features.len());
        for (stm, ntm) in stm_ti_features.into_iter().zip(ntm_ti_features) {
            ti(stm.cast::<usize>(), ntm.cast::<usize>());
        }

        let kafts = [White, Black].map(|side| KAFeature::lut(side, ksqs[side], &pos).to_array());

        for sq in Bitboard::from(pos.occupied()).iter() {
            let kaft = kafts.map(|ft| <KAFeature as Num>::new(ft[sq]));
            ka(kaft[White].cast::<usize>(), kaft[Black].cast::<usize>());
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[rustfmt::skip]
type Inputs = (((((((SparseInput, SparseInput), SparseInput), SparseInput), SparseInput), SparseInput), SparseInput), DenseInput<f32>);

fn make_mapper(
    inputs: &ModelInputs<Inputs>,
    features: Features,
    phaser: Phaser,
    wdl: impl WdlScheduler,
) -> ModelInputsMapper<ChessBoard> {
    ModelInputsMapper::build(inputs, move |pos, step, inputs| {
        let (((((((stm_pp, ntm_pp), stm_ti), ntm_ti), stm_ka), ntm_ka), phase), target) = inputs;

        stm_pp.fill(-1);
        ntm_pp.fill(-1);
        stm_ti.fill(-1);
        ntm_ti.fill(-1);
        stm_ka.fill(-1);
        ntm_ka.fill(-1);

        let mut pp_cnt = 0;
        let mut ti_cnt = 0;
        let mut ka_cnt = 0;
        features.map_features(
            *pos,
            |s, n| {
                stm_pp[pp_cnt] = s as i32;
                ntm_pp[pp_cnt] = n as i32;
                pp_cnt += 1;
            },
            |s, n| {
                stm_ti[ti_cnt] = s as i32;
                ntm_ti[ti_cnt] = n as i32;
                ti_cnt += 1;
            },
            |s, n| {
                stm_ka[ka_cnt] = s as i32;
                ntm_ka[ka_cnt] = n as i32;
                ka_cnt += 1;
            },
        );

        assert!(pp_cnt <= PPFeature::MAX_ACTIVE);
        assert!(ti_cnt <= TIFeature::MAX_ACTIVE);
        assert!(ka_cnt <= KAFeature::MAX_ACTIVE);

        phase[0] = phaser.bucket(pos).cast();

        let score = f32::from(pos.score);
        let p = (score - TARGET_SCORE_OFFSET) / TARGET_SCORE_SCALING;
        let pm = (-score - TARGET_SCORE_OFFSET) / TARGET_SCORE_SCALING;
        let score_wdl = 0.5 * (1.0 + sigmoid(p) - sigmoid(pm));

        let result = f32::from(pos.result) / 2.0;
        let blend = wdl.blend(step.batch(), step.superbatch(), step.final_superbatch());
        target[0] = blend * result + (1.0 - blend) * score_wdl;
    })
}

#[derive(Debug, Clone, Copy)]
struct Offender {
    bullet: f32,
    engine: f32,
    board: Board,
}

impl Offender {
    fn error(&self) -> f32 {
        self.bullet - self.engine
    }

    fn relative_error(&self) -> f32 {
        self.error() / (self.bullet.abs() + 0.1)
    }
}

impl Eq for Offender {}

impl PartialEq for Offender {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Ord for Offender {
    fn cmp(&self, other: &Self) -> Ordering {
        self.relative_error().total_cmp(&other.relative_error())
    }
}

impl PartialOrd for Offender {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct ErrorStats {
    count: usize,
    error_squared: f32,
    error_absolute: f32,
    target_absolute: f32,
}

impl ErrorStats {
    fn add(&mut self, target: f32, inferred: f32) {
        let error = target - inferred;
        self.error_squared += error.powi(2);
        self.error_absolute += error.abs();
        self.target_absolute += target.abs();
        self.count += 1;
    }

    fn rmse(&self) -> f32 {
        if self.count == 0 || self.target_absolute == 0.0 {
            0.0
        } else {
            let error = self.error_squared / self.count as f32;
            let target = self.target_absolute / self.count as f32;
            error.sqrt() / target
        }
    }

    fn mae(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.error_absolute / self.count as f32
        }
    }
}

#[derive(Deref)]
struct Trainer<'a> {
    #[deref]
    orchestrator: &'a Orchestrator,
    optimiser: Optimiser<ExecutionContext, AdamW<ExecutionContext>>,
    inputs: ModelInputs<Inputs>,
    features: Features,
    phaser: Phaser,
    saved_format: Vec<SavedFormat>,
}

impl<'a> Trainer<'a> {
    fn new(orchestrator: &'a Orchestrator) -> Result<Self, Failure> {
        let phaser = Phaser;
        let features = Features;
        let inputs = ModelInputs::default()
            .add_sparse("stm/pp", (PPFeature::LEN, 1), PPFeature::MAX_ACTIVE)
            .add_sparse("ntm/pp", (PPFeature::LEN, 1), PPFeature::MAX_ACTIVE)
            .add_sparse("stm/ti", (TIFeature::LEN, 1), TIFeature::MAX_ACTIVE)
            .add_sparse("ntm/ti", (TIFeature::LEN, 1), TIFeature::MAX_ACTIVE)
            .add_sparse("stm/ka", (KAFeature::LEN, 1), KAFeature::MAX_ACTIVE)
            .add_sparse("ntm/ka", (KAFeature::LEN, 1), KAFeature::MAX_ACTIVE)
            .add_sparse("buckets", (Phase::LEN, 1), 1)
            .add_dense("targets", (1, 1));

        let model = ModelDefinition::build(&inputs, |builder, input| {
            let (((((((stm_pp, ntm_pp), stm_ti), ntm_ti), stm_ka), ntm_ka), phase), target) = input;

            let normal_init = InitSettings::Normal {
                stdev: (2.0 / Features::MAX_ACTIVE as f32).sqrt(),
                mean: 0.0,
            };

            let pp_shape = Shape::new(Accumulator::LEN, PPFeature::LEN);
            let pp = builder.new_weights("pp/w", pp_shape, normal_init.clone());
            let pp = pp.faux_quantise(FTQ.cast(), true);

            let ti_shape = Shape::new(Accumulator::LEN, TIFeature::LEN);
            let ti = builder.new_weights("ti/w", ti_shape, normal_init.clone());
            let ti = ti.faux_quantise(FTQ.cast(), true);

            let psq_shape = Shape::new(Accumulator::LEN, PSQFeature::LEN);
            let psq = builder.new_weights("ka/psq", psq_shape, InitSettings::Zeroed);

            let mut ka = builder.new_affine("ka/", KAFeature::LEN, Accumulator::LEN);
            ka.init_with_effective_input_size(Features::MAX_ACTIVE);

            ka.weights = ka.weights + psq.repeat(Features::BUCKETS);
            ka.weights = ka.weights.clip_pass_through_grad(-FT_CLIP, FT_CLIP);
            ka.weights = ka.weights.faux_quantise(FTQ.cast(), true);
            ka.bias = ka.bias.faux_quantise(FTQ.cast(), true);

            let ft = |pp_in, ti_in, ka_in, start, end| {
                let pp = pp.slice_rows(start, end).matmul(pp_in);
                let ti = ti.slice_rows(start, end).matmul(ti_in);
                let ka = ka.slice(start, end).forward(ka_in);
                (pp + ti + ka).crelu()
            };

            let mut l12 = builder.new_affine("l12/", Li::LEN, Phase::LEN * Ln::LEN / 2);
            l12.weights = l12.weights.faux_quantise(HLQ.cast(), true);

            let l23 = builder.new_affine("l23/", Ln::LEN, Phase::LEN * Ln::LEN / 2);
            let l34 = builder.new_affine("l34/", Ln::LEN, Phase::LEN * Ln::LEN / 2);
            let l4o = builder.new_affine("l4o/", Ln::LEN, Phase::LEN);

            let shape = Shape::new(Phase::LEN, Ln::LEN);
            let r2o = builder.new_weights("r2o/w", shape, InitSettings::Zeroed);

            let stm_lo = ft(stm_pp, stm_ti, stm_ka, 0, Li::LEN / 2);
            let stm_hi = ft(stm_pp, stm_ti, stm_ka, Li::LEN / 2, Li::LEN);
            let ntm_lo = ft(ntm_pp, ntm_ti, ntm_ka, 0, Li::LEN / 2);
            let ntm_hi = ft(ntm_pp, ntm_ti, ntm_ka, Li::LEN / 2, Li::LEN);

            let stm = (stm_lo * stm_hi).faux_quantise(1.0 / (I2F * HLQ as f32), true);
            let ntm = (ntm_lo * ntm_hi).faux_quantise(1.0 / (I2F * HLQ as f32), true);

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

            let score = F2V * out;
            let qp = (score - INFERRED_SCORE_OFFSET) / INFERRED_SCORE_SCALING;
            let qn = (-score - INFERRED_SCORE_OFFSET) / INFERRED_SCORE_SCALING;
            let inferred = 0.5 * (1.0 + qp.sigmoid() - qn.sigmoid());

            let err = inferred - target;
            let err_relu = err.relu();
            let loss = err * err + 0.15 * err_relu * err_relu + 0.005 * l1_reg;

            (
                Some(loss.reduce_sum_batch()),
                vec![("output".to_string(), out)],
            )
        });

        let saved_format = vec![
            SavedFormat::id("pp/w").round().quantise::<i8>(FTQ),
            SavedFormat::id("ti/w").round().quantise::<i8>(FTQ),
            SavedFormat::id("ka/w")
                .transform(move |store, mut values| {
                    let psq = store.get("ka/psq").values.f32().repeat(Features::BUCKETS);
                    for (v, f) in values.iter_mut().zip(psq) {
                        *v = (*v + f).clamp(-FT_CLIP, FT_CLIP);
                    }

                    values
                })
                .round()
                .quantise::<i8>(FTQ),
            SavedFormat::id("ka/b").round().quantise::<i16>(FTQ),
            SavedFormat::id("l12/w")
                .transpose()
                .round()
                .quantise::<i8>(HLQ),
            SavedFormat::id("l12/b"),
            SavedFormat::id("r2o/w").transpose(),
            SavedFormat::id("l23/w").transpose(),
            SavedFormat::id("l23/b"),
            SavedFormat::id("l34/w").transpose(),
            SavedFormat::id("l34/b"),
            SavedFormat::id("l4o/w").transpose(),
            SavedFormat::id("l4o/b"),
        ];

        let device = DefaultDevice::new(0).or_bail()?;
        let weights = ModelWeights::new(&model, 0x111D59E599BE370C);
        let params = AdamWParams::default();
        let mut optimiser = Optimiser::new(model, weights, device, params).or_bail()?;

        optimiser.set_params(AdamWParams {
            min_weight: f32::MIN,
            max_weight: f32::MAX,
            ..AdamWParams::default()
        });

        for id in ["pp/w", "ti/w", "ka/w", "ka/psq"] {
            optimiser.set_params_for_weight(
                id,
                AdamWParams {
                    min_weight: -FT_CLIP,
                    max_weight: FT_CLIP,
                    ..AdamWParams::default()
                },
            );
        }

        optimiser.set_params_for_weight(
            "l12/w",
            AdamWParams {
                min_weight: -HL_CLIP,
                max_weight: HL_CLIP,
                ..AdamWParams::default()
            },
        );

        Ok(Self {
            optimiser,
            inputs,
            features,
            phaser,
            saved_format,
            orchestrator,
        })
    }

    fn load_from_checkpoint(&mut self, stage: usize, superbatch: usize) -> Result<(), Failure> {
        let checkpoint = format!(
            "{}/stage{stage}-{superbatch}/optimiser_state",
            self.checkpoints,
        );

        self.optimiser.load_from_checkpoint(&checkpoint).or_bail()
    }

    fn eval(&mut self, positions: usize, worst: usize, settled_score: f32) -> Result<(), Failure> {
        let device = self.optimiser.device();
        let model = self.optimiser.definition();
        let weights = self.optimiser.weights();

        let mut evaluator = ModelEvaluator::new(model, device.clone()).or_bail()?;
        evaluator.load_device_weights(weights).or_bail()?;

        let mapper = make_mapper(
            &self.inputs,
            self.features,
            self.phaser,
            ConstantWDL { value: 0.0 },
        );

        let reader = SfBinpackLoader::new_concat_multiple(
            &[self.dataset.as_str()],
            self.buffer_size,
            self.threads,
            |_| true,
        );

        let mut boards = Vec::with_capacity(positions);
        reader.read_chunks(0, |chunk| {
            let rest = positions.saturating_sub(boards.len());
            let take = rest.min(chunk.len());
            boards.extend_from_slice(&chunk[..take]);
            boards.len() >= positions
        });

        if boards.len() < positions {
            bail!("unexpected EOF after {} < {positions}", boards.len());
        }

        let mut stats = ErrorStats::default();
        let mut clipped_stats = ErrorStats::default();
        let mut unsettled_stats = ErrorStats::default();
        let mut settled_stats = ErrorStats::default();
        let mut offenders = BinaryHeap::with_capacity(worst + 1);

        for board in &boards {
            let batch = mapper.map(slice::from_ref(board), Step::default(), 1);
            let inputs = batch.to_device(&device).or_bail()?;
            let output = evaluator.evaluate(&inputs).or_bail()?.get("output");
            let output = output.ok_or_else(|| Failure::msg("missing `output` node"))?;
            let [bullet] = output.to_host().or_bail()?.f32()[..] else {
                bail!("`output` is not an f32 scalar");
            };

            let bullet = bullet * F2V;
            let engine = Evaluator::from(*board).evaluate();
            stats.add(bullet, engine);

            clipped_stats.add(
                bullet.clamp(-settled_score, settled_score),
                engine.clamp(-settled_score, settled_score),
            );

            if f32::from(board.score).abs() >= settled_score {
                settled_stats.add(bullet, engine);
            } else {
                unsettled_stats.add(bullet, engine);
                offenders.push(Reverse(Offender {
                    board: Board::from(*board),
                    engine,
                    bullet,
                }));
            }

            if offenders.len() > worst {
                offenders.pop();
            }
        }

        println!(
            "unclipped RMSE {:.2}% and MAE {:.2} cp over {} positions",
            100.0 * stats.rmse(),
            stats.mae(),
            stats.count
        );

        if clipped_stats.count > 0 {
            println!(
                "clipped({}) RMSE {:.2}% and MAE {:.2} cp over {} positions",
                settled_score,
                100.0 * clipped_stats.rmse(),
                clipped_stats.mae(),
                clipped_stats.count
            );
        }

        if unsettled_stats.count > 0 {
            println!(
                "unsettled(< {}) RMSE {:.2}% and MAE {:.2} cp over {} positions",
                settled_score,
                100.0 * unsettled_stats.rmse(),
                unsettled_stats.mae(),
                unsettled_stats.count
            );
        }

        if settled_stats.count > 0 {
            println!(
                "settled(>= {}) RMSE {:.2}% and MAE {:.2} cp over {} positions",
                settled_score,
                100.0 * settled_stats.rmse(),
                settled_stats.mae(),
                settled_stats.count
            );
        }

        if !offenders.is_empty() {
            println!("worst {} offenders:", offenders.len());
            for Reverse(offender) in offenders.into_sorted_vec() {
                println!("  {}", offender.board,);
                println!(
                    "    bullet={:.2} engine={:.2} error={:.2}%",
                    offender.bullet,
                    offender.engine,
                    100.0 * offender.relative_error()
                );
            }
        }

        Ok(())
    }

    fn run(
        &mut self,
        id: &str,
        sb: RangeInclusive<usize>,
        wdl: RangeInclusive<f32>,
        lr: impl LrScheduler,
    ) -> Result<(), Failure> {
        let (start_superbatch, end_superbatch) = sb.into_inner();
        let (start_wdl, end_wdl) = wdl.into_inner();

        let mapper = make_mapper(
            &self.inputs,
            self.features,
            self.phaser,
            LinearWDL {
                start: start_wdl,
                end: end_wdl,
            },
        );

        let filter = self.filter.clone();
        let reader = SfBinpackLoader::new_concat_multiple(
            &[self.dataset.as_str()],
            self.buffer_size,
            self.threads,
            move |entry| !filter.should_skip(entry),
        );

        let schedule = TrainingSchedule {
            log_rate: self.log_rate,
            lr_schedule: lr.boxed(),
            steps: TrainingSteps {
                batch_size: self.batch_size,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch,
                end_superbatch,
            },
        };

        let save_rate = self.save_rate;
        let checkpoints = self.orchestrator.checkpoints.clone();
        let dataloader = ReadMapLoader::new(reader, mapper, self.threads.saturate());

        let mut ticks = 0.0f32;
        let mut running_loss = 0.0f32;
        let error = RefCell::new(Vec::new());

        let result = train(
            &mut self.optimiser,
            schedule,
            dataloader,
            |_, step, loss| {
                running_loss += loss;
                ticks += 1.0;

                if step.batch().is_multiple_of(32)
                    || (step.batches_per_superbatch() < 32
                        && step.batch() == step.batches_per_superbatch())
                {
                    error.borrow_mut().push((
                        step.superbatch(),
                        step.batch(),
                        running_loss / ticks.min(step.batches_per_superbatch().cast()),
                    ));

                    running_loss = 0.0;
                    ticks = 0.0;
                }
            },
            |optimiser, step| {
                let sb = step.superbatch();
                if sb % save_rate == 0 || sb == step.final_superbatch() {
                    let prefix = format!("{}/{}-{}", checkpoints, id, sb);
                    save_to_checkpoint(optimiser, &self.saved_format, &prefix);
                    write_losses(&format!("{prefix}/log.txt"), &error.borrow());
                }
            },
        );

        result.or_bail()
    }

    fn train(&mut self, stage: usize, superbatch: usize) -> Result<(), Failure> {
        create_dir_all(&self.checkpoints)?;

        if stage == 0 && superbatch < SB0 {
            const WARMUP_SBS: usize = SB0 / 2;
            const COOLDOWN_SBS: usize = SB0 - WARMUP_SBS;

            let lr = Sequence {
                first: LinearDecayLR {
                    initial_lr: 1e-4,
                    final_lr: 5e-3,
                    final_superbatch: WARMUP_SBS,
                },
                second: LinearDecayLR {
                    initial_lr: 5e-3,
                    final_lr: 1e-4,
                    final_superbatch: COOLDOWN_SBS,
                },
                first_scheduler_final_superbatch: WARMUP_SBS,
            };

            let start = if stage == 0 { superbatch + 1 } else { 1 };
            self.run("stage0", start..=SB0, 0.0..=0.0, lr)?;
        }

        if stage < 1 || (stage == 1 && superbatch < SB1) {
            let lr = LinearDecayLR {
                initial_lr: 1e-3,
                final_lr: 1e-6,
                final_superbatch: SB1,
            };

            let start = if stage == 1 { superbatch + 1 } else { 1 };
            self.run("stage1", start..=SB1, 0.0..=self.wdl, lr)?;
        }

        if stage < 2 || (stage == 2 && superbatch < SB2) {
            let lr = LinearDecayLR {
                initial_lr: 1e-6,
                final_lr: 1e-7,
                final_superbatch: SB2,
            };

            let start = if stage == 2 { superbatch + 1 } else { 1 };
            self.run("stage2", start..=SB2, self.wdl..=self.wdl, lr)?;
        }

        Ok(())
    }
}

/// An efficiently updatable neural network (NNUE) trainer.
#[derive(Debug, Parser)]
struct Orchestrator {
    /// How many threads to use for data loading.
    #[clap(long, default_value_t = available_parallelism().map_or(1, NonZero::get).div(2).max(1))]
    threads: usize,

    /// How many positions per batch.
    #[clap(long, default_value_t = 131072)]
    batch_size: usize,

    /// How many batches per superbatch.
    #[clap(long, default_value_t = 768)]
    batches_per_superbatch: usize,

    /// Data loader buffer size in MB.
    #[clap(long, default_value_t = 1024)]
    buffer_size: usize,

    /// How often to log progress, in batches.
    #[clap(long, default_value_t = 16)]
    log_rate: usize,

    /// How often to write checkpoints, in superbatches.
    #[clap(long, default_value_t = 20)]
    save_rate: usize,

    /// The path where to store checkpoints.
    #[clap(long, default_value = "checkpoints/")]
    checkpoints: String,

    /// The target wdl fraction.
    #[clap(long, default_value_t = 0.0)]
    wdl: f32,

    #[clap(flatten)]
    filter: TrainingDataFilter,

    /// The datasets to use for training.
    #[clap(long)]
    dataset: String,

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

    /// Evaluates dataset positions using weights in a checkpoint.
    Eval {
        /// The stage to load from
        #[clap(long)]
        stage: usize,
        /// The superbatch to load from
        #[clap(long)]
        superbatch: usize,
        /// How many dataset positions to evaluate.
        #[clap(long, default_value_t = 5000)]
        positions: usize,
        /// How many worst offenders to report.
        #[clap(long, default_value_t = 10)]
        worst: usize,
        /// Score beyond which to assume the outcome is settled.
        #[clap(long, default_value_t = 1200.0)]
        settled_score: f32,
    },
}

impl Orchestrator {
    fn run(&self) -> Result<(), Failure> {
        let mut trainer = Trainer::new(self)?;

        match &self.mode {
            Mode::Start => trainer.train(0, 0),
            Mode::Resume { stage, superbatch } => {
                trainer.load_from_checkpoint(*stage, *superbatch)?;
                trainer.train(*stage, *superbatch)
            }

            Mode::Eval {
                stage,
                superbatch,
                positions,
                worst,
                settled_score,
            } => {
                trainer.load_from_checkpoint(*stage, *superbatch)?;
                trainer.eval(*positions, *worst, *settled_score)
            }
        }
    }
}

fn main() -> Result<(), Failure> {
    Orchestrator::parse().run()
}

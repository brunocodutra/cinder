use crate::util::*;
use std::{cell::SyncUnsafeCell, ops::Deref, slice::SliceIndex};

#[cfg(feature = "spsa")]
use derive_more::with_trait::Display;

#[cfg(feature = "spsa")]
use ron::{Error, de::SpannedError, from_str as deserialize, ser::to_writer as serialize};

#[cfg(feature = "spsa")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "spsa")]
use std::fmt::{self, Formatter};

#[cfg(feature = "spsa")]
use std::str::FromStr;

#[cfg(test)]
#[cfg(feature = "spsa")]
use proptest::{collection::vec, prelude::*};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "spsa", serde(into = "Box<[f32]>", try_from = "Box<[f32]>"))]
struct Param<const BYTES: ConstBytes<32>> {
    #[cfg(feature = "spsa")]
    #[cfg_attr(test, strategy(vec(-1e3f32..=1e3f32, Self::VALUES.len().cast::<usize>())
        .prop_map(|vs| Seq::from_iter(vs))))]
    values: ConstSeq<f32, 32>,
}

impl<const BYTES: ConstBytes<32>> Param<BYTES> {
    const VALUES: &ConstSeq<f32, 32> = unsafe { &Seq::reify(BYTES) };

    const fn new() -> Self {
        Self {
            #[cfg(feature = "spsa")]
            values: Self::VALUES.clone(),
        }
    }

    #[cfg(feature = "spsa")]
    fn perturb<I: IntoIterator<Item = f32>>(&self, perturbations: I) -> (Self, Self) {
        let (mut left, mut right) = (self.clone(), self.clone());
        let mut perturbations = perturbations.into_iter();
        for (i, c) in Self::VALUES.iter().enumerate() {
            let delta = c.abs() * perturbations.next().unwrap();
            left.values[i] += delta;
            right.values[i] -= delta;
        }

        (left, right)
    }

    #[cfg(feature = "spsa")]
    fn update<I: IntoIterator<Item = f32>>(&mut self, corrections: I) {
        let mut corrections = corrections.into_iter();
        for (i, c) in Self::VALUES.iter().enumerate() {
            let delta = c.abs() * corrections.next().unwrap();
            self.values[i] += delta;
        }
    }
}

impl<const BYTES: ConstBytes<32>> const Default for Param<BYTES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const BYTES: ConstBytes<32>> const Deref for Param<BYTES> {
    type Target = [f32];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        #[cfg(feature = "spsa")]
        {
            &self.values
        }

        #[cfg(not(feature = "spsa"))]
        {
            Self::VALUES
        }
    }
}

#[cfg(feature = "spsa")]
impl<const BYTES: ConstBytes<32>> From<Param<BYTES>> for Box<[f32]> {
    fn from(param: Param<BYTES>) -> Self {
        param.values.into_iter().collect()
    }
}

#[cfg(feature = "spsa")]
impl<const BYTES: ConstBytes<32>> TryFrom<Box<[f32]>> for Param<BYTES> {
    type Error = Error;

    fn try_from(values: Box<[f32]>) -> Result<Self, Self::Error> {
        if values.len() <= Self::VALUES.len().cast() {
            Ok(Param {
                values: values.into_iter().collect(),
            })
        } else {
            Err(Error::ExpectedDifferentLength {
                expected: Self::VALUES.len().to_string(),
                found: values.len(),
            })
        }
    }
}

static PARAMS: SyncUnsafeCell<Params> = SyncUnsafeCell::new(Params::new());

#[cfg(feature = "spsa")]
impl Display for Params {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        #[expect(clippy::map_err_ignore)]
        serialize(f, self).map_err(|_| fmt::Error)
    }
}

#[cfg(feature = "spsa")]
impl FromStr for Params {
    type Err = SpannedError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        deserialize(s)
    }
}

#[cfg(feature = "spsa")]
macro_rules! len {
    ($head:expr,) => { $head.len() };
    ($head:expr, $($tail:expr,)*) => {
        $head.len() + len!($($tail,)*)
    }
}

macro_rules! params {
    ($($name: ident: $value: expr,)*) => {
        #[derive(Debug, Default, Clone, PartialEq)]
        #[cfg_attr(test, derive(test_strategy::Arbitrary))]
        #[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
        #[cfg_attr(feature = "spsa", serde(deny_unknown_fields))]
        pub struct Params {
            $(
                #[cfg_attr(feature = "spsa", serde(default))]
                $name: Param<{ Bytes::from($value as [f32; _]) }>,
            )*
        }

        impl Params {
            const fn new() -> Self {
                Params {
                    $($name: Param::new(),)*
                }
            }
        }

        $(impl Params {
            /// This parameter's current value.
            #[inline(always)]
            pub const fn $name<R: [const] SliceIndex<[f32]>>(idx: R) -> &'static R::Output {
                unsafe { PARAMS.get().as_ref_unchecked().$name.get_unchecked(idx) }
            }
        })*

        #[cfg(feature = "spsa")]
        impl Params {
            /// Total number of parameters.
            pub const LEN: usize = len!($($value,)*);

            /// Initializes the global params.
            pub fn init(self) {
                unsafe { *PARAMS.get().as_mut_unchecked() = self }
            }

            /// Perturb parameters in both positive and negative directions.
            ///
            /// # Panic
            ///
            /// Panics if the `perturbations`'s length is less than [`Self::LEN`].
            pub fn perturb<I: IntoIterator<Item = f32>>(&self, perturbations: I) -> (Self, Self) {
                let mut perturbations = perturbations.into_iter();
                let (mut left, mut right) = (Self::default(), Self::default());
                $((left.$name, right.$name) = self.$name.perturb(&mut perturbations);)*
                (left, right)
            }

            /// Update parameters in-place.
            ///
            /// # Panic
            ///
            /// Panics if the `corrections`'s length is less than [`Self::LEN`].
            pub fn update<I: IntoIterator<Item = f32>>(&mut self, corrections: I) {
                let mut corrections = corrections.into_iter();
                $(self.$name.update(&mut corrections);)*
            }
        }
    };
}

params! {
    moves_left_start: [180.31662],
    moves_left_end: [2.0940008],
    soft_time_fraction: [0.7614044],
    hard_time_fraction: [0.8543253],
    score_trend_inertia: [0.12636957],
    pv_focus_gamma: [-1.6757277],
    pv_focus_delta: [1.879524],
    score_trend_magnitude: [0.87890685],
    score_trend_pivot: [27.931425],
    improving: [0.0, 0.88847524, 0.9035022, 1.8151929],
    pawn_values: [40.68328, 51.843765, 51.668552, 57.464756, 58.076283, 58.282803, 57.37344, 51.2746],
    knight_values: [236.87602, 186.71053, 196.50052, 188.28653, 201.66992, 184.64095, 177.26595, 169.36464],
    bishop_values: [243.5236, 209.7292, 213.33699, 222.99481, 212.59624, 215.26357, 208.33934, 193.58437],
    rook_values: [248.23985, 317.3855, 327.00903, 332.0516, 330.6984, 319.5671, 320.80914, 293.0956],
    queen_values: [541.7995, 589.19336, 640.0253, 661.2913, 601.1991, 641.9814, 612.8087, 646.1975],
    pawns_correction: [14.037831, 13.509618, 13.724369, 13.832029, 13.668373, 13.740679, 14.594005, 13.493925],
    minor_correction: [9.8398695, 10.001816, 9.820436, 9.79345, 9.887946, 10.301183, 9.605013, 9.68262],
    major_correction: [11.674672, 10.629728, 10.993739, 11.263099, 11.282832, 11.086999, 10.958088, 11.541785],
    pieces_correction: [16.361282, 15.755122, 16.123756, 16.472897, 16.809092, 16.7554, 16.481047, 16.453663],
    pawns_correction_bonus: [1.0098459],
    minor_correction_bonus: [0.9991274],
    major_correction_bonus: [1.0689667],
    pieces_correction_bonus: [1.0677278],
    history_bonus_depth: [0.0, 5.1406612],
    history_bonus_scalar: [0.73648643],
    continuation_bonus_depth: [0.0, 6.761449],
    continuation_bonus_scalar: [0.8777971],
    history_penalty_depth: [0.0, -5.9487557],
    history_penalty_scalar: [-1.5699182],
    continuation_penalty_depth: [0.0, -5.4533486],
    continuation_penalty_scalar: [-0.72453064],
    probcut_margin_depth: [0.0, 13.675054],
    probcut_margin_scalar: [203.5154],
    single_extension_margin_depth: [0.0, 0.7130737],
    single_extension_margin_scalar: [0.55041635],
    double_extension_margin_depth: [0.0, 1.0597025],
    double_extension_margin_scalar: [0.26570204],
    triple_extension_margin_depth: [0.0, 0.5865015],
    triple_extension_margin_scalar: [144.4708],
    tt_cut_halfmove_limit: [87.07181],
    tb_cut_depth_bonus: [4.020102],
    flp_margin_depth: [23.94976, 332.64258],
    flp_margin_scalar: [-130.9923],
    fhp_margin_depth: [31.863237, 63.602863],
    fhp_margin_scalar: [-50.49814],
    nmp_margin_depth: [0.5478835, 9.937967],
    nmp_margin_scalar: [-5.015432],
    nmr_gamma: [0.12773624],
    nmr_delta: [0.118220635],
    nmr_limit: [3.0748518],
    nmr_fraction: [0.25809592],
    razoring_depth: [8.300784, 21.358389],
    razoring_scalar: [32.38831],
    rfp_margin_depth: [1.2257179, 12.044115],
    rfp_margin_scalar: [9.293389],
    rfp_margin_improving: [-5.8120627],
    fut_margin_depth: [1.7254328, 25.66078],
    fut_margin_scalar: [52.251163],
    fut_margin_is_pv: [7.104991],
    fut_margin_was_pv: [5.867695],
    fut_margin_is_check: [7.780886],
    fut_margin_is_killer: [6.7144685],
    fut_margin_improving: [4.306905],
    fut_margin_gain: [1.2003005],
    nsp_margin_depth: [-2.1436746, -47.1968],
    nsp_margin_scalar: [8.411735],
    qsp_margin_depth: [-9.15234, -0.87013435],
    qsp_margin_scalar: [8.525748],
    sp_margin_is_killer: [-7.0181723],
    lmp_depth: [0.4447052, 0.31299993],
    lmp_scalar: [0.9277796],
    lmp_baseline: [0.7069476],
    lmp_is_root: [1.6324568],
    lmp_is_pv: [1.1405352],
    lmp_was_pv: [0.7119965],
    lmp_is_check: [0.9584817],
    lmp_improving: [0.9374783],
    lmr_depth: [0.0, 0.22966103, 0.0985384],
    lmr_index: [0.0, 0.10215471],
    lmr_scalar: [0.74981624],
    lmr_baseline: [0.31234685],
    lmr_is_root: [-0.26101434],
    lmr_is_pv: [-0.7487195],
    lmr_was_pv: [-0.27339074],
    lmr_gives_check: [-0.9181918],
    lmr_is_noisy_pv: [0.7140092],
    lmr_is_killer: [-1.2464437],
    lmr_cut: [1.2892803],
    lmr_improving: [-0.37845215],
    lmr_history: [-0.84320587],
    lmr_counter: [-1.2009448],
    killer_rating: [51.94935],
    history_rating: [106.13061],
    counter_rating: [125.73856],
    winning_rating_margin: [-20.679895],
    winning_rating_gain: [0.0, 1.6378232],
    winning_rating_scalar: [15.620518],
    aw_baseline: [2041.4463, 514.4356, 129.16896, 31.456507, 7.939967, 4.827342, 4.900985, 4.8683586],
    aw_gamma: [1.4410394],
    aw_delta: [1.237055],
    aw_fail_low_blend: [0.5187618],
    aw_fail_high_reduction: [1.0361685],
}

#[cfg(test)]
#[cfg(feature = "spsa")]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn perturbing_updates_params(
        p: Params,
        #[strategy(vec(-1f32..=1f32, Params::LEN))] d: Vec<f32>,
    ) {
        let (mut l, mut r) = (p.clone(), p.clone());
        l.update(d.iter().copied());
        r.update(d.iter().map(|&d| -d));
        assert_eq!(p.perturb(d), (l, r));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_params_is_an_identity(p: Params) {
        assert_eq!(p.to_string().parse(), Ok(p));
    }
}

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
struct Param<const BYTES: ConstBytes<64>> {
    #[cfg(feature = "spsa")]
    #[cfg_attr(test, strategy(vec(-1e3f32..=1e3f32, Self::VALUES.len().cast::<usize>()).prop_map(Seq::from_iter)))]
    values: ConstSeq<f32, 64>,
}

impl<const BYTES: ConstBytes<64>> Param<BYTES> {
    const VALUES: &ConstSeq<f32, 64> = unsafe { &Seq::reify(BYTES) };

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

impl<const BYTES: ConstBytes<64>> const Default for Param<BYTES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const BYTES: ConstBytes<64>> const Deref for Param<BYTES> {
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
impl<const BYTES: ConstBytes<64>> From<Param<BYTES>> for Box<[f32]> {
    fn from(param: Param<BYTES>) -> Self {
        param.values.into_iter().collect()
    }
}

#[cfg(feature = "spsa")]
impl<const BYTES: ConstBytes<64>> TryFrom<Box<[f32]>> for Param<BYTES> {
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
    moves_left_start: [160.94777],
    moves_left_end: [0.483503],
    moves_left_damping: [1.004061],
    soft_time_fraction: [0.7399792],
    hard_time_fraction: [0.86297363],
    score_trend_inertia: [0.120955214],
    pv_focus_gamma: [-1.6761804],
    pv_focus_delta: [1.8973947],
    score_trend_magnitude: [0.86272377],
    score_trend_pivot: [27.789158],
    improving: [0.0, 0.88212675, 0.91403663, 1.8949512],
    piece_values: [52.847713, 188.47716, 215.22618, 306.7143, 629.2747, 0.0],
    pawns_correction: [13.677705],
    minor_correction: [9.955085],
    major_correction: [10.968398],
    pieces_correction: [16.386993],
    continuation_correction: [12.0, 8.0],
    pawns_correction_delta: [0.00059015466, -0.25, 0.25],
    minor_correction_delta: [0.00057724636, -0.25, 0.25],
    major_correction_delta: [0.0006286239, -0.25, 0.25],
    pieces_correction_delta: [0.0006316802, -0.25, 0.25],
    counter_correction_delta: [0.0006, -0.25, 0.25],
    followup_correction_delta: [0.0006, -0.25, 0.25],
    history_bonus: [0.04, 0.0028119748, 0.25],
    history_penalty: [-0.045, -0.006025375, -0.25],
    counter_bonus: [0.052, 0.0034933158, 0.25],
    counter_penalty: [-0.042, -0.00282255, -0.25],
    followup_bonus: [0.052, 0.0034933158, 0.25],
    followup_penalty: [-0.042, -0.00282255, -0.25],
    probcut_margin_depth: [0.0, 13.350125],
    probcut_margin_scalar: [209.42744],
    probcut_depth: [0.0, 3.0],
    probcut_depth_bounds: [6.0, 3.0],
    singular_margin_depth: [0.0, 0.4],
    singular_margin_scalar: [0.4, 6.0, 24.0],
    singular_depth: [0.5, -0.5],
    singular_depth_bounds: [6.0, 3.0],
    tt_cut_halfmove_limit: [87.07959],
    tb_cut_depth_bonus: [3.8686926],
    flp_margin_depth: [24.092096, 343.5552],
    flp_margin_scalar: [-132.51001],
    fhp_margin_depth: [32.207886, 63.049282],
    fhp_margin_scalar: [-47.862656],
    nmp_margin_depth: [0.55535954, 10.07757],
    nmp_margin_scalar: [-4.991574],
    nmr_depth: [0.3055916],
    nmr_score: [0.1415365, 0.1190361, 3.447467],
    razoring_alpha_limit: [2000.0],
    razoring_depth: [8.4302845, 20.976559],
    razoring_scalar: [30.856945],
    rfp_margin_depth: [1.238703, 11.705757],
    rfp_margin_scalar: [9.479503],
    rfp_margin_improving: [-5.70555],
    fut_depth_limit: [16.0],
    futility_margin_depth: [4.0, 12.0],
    futility_margin_scalar: [60.0],
    futility_margin_quiescence: [60.0],
    nsp_margin_depth: [-2.1050088, -47.118065],
    nsp_margin_scalar: [8.325452],
    qsp_margin_depth: [-8.862688, -0.8800826],
    qsp_margin_scalar: [8.587346],
    qsp_margin_is_killer: [-7.17011],
    lmp_depth: [0.3075955, 0.2199919],
    lmp_scalar: [0.6577168],
    lmp_improving: [1.3111653],
    lmr_index: [0.0, 0.26877472, 0.04644205],
    lmr_depth: [0.0, 0.050253514],
    lmr_scalar: [0.53588414],
    lmr_is_root: [-0.25634402, 0.347708, -0.44826812, -0.8956343],
    lmr_not_root: [0.3236711, -0.34688452, -0.13890503, 0.59107697, -0.39018932, -0.6227634, 0.33730632, -0.44872612, -0.93183327, -1.2027539],
    lmr_is_pv: [-0.3742653, -0.0040767305, 0.023327855, -0.010980638, 0.01476776, -0.008847479, -0.0056572794, 0.0164216, -0.00018493863],
    lmr_was_pv: [-0.13503297, -0.0012335496, 0.0021022086, 0.0010632597, 0.007853777, 0.014211153, -0.002537641, -0.0015321401],
    lmr_is_cut: [0.6413949, -0.0025511023, -0.0038119762, 0.009320957, 0.006868088, -0.0031091466, -0.019076373],
    lmr_improving: [0.0070772306, -0.00502337, 0.0041178335, -0.004997687, -0.005575128, 0.00234888],
    lmr_is_killer: [-0.62619746, 0.003998975, -0.001831985, 0.0085495375, -0.00048835564],
    lmr_is_noisy_pv: [0.35211658, 0.004617254, 0.008945197, -0.003098197],
    lmr_gives_check: [-0.46384782, 0.015132483, 0.011271138],
    lmr_history: [0.0012702212, -0.003735534],
    lmr_counter: [0.008251999],
    killer_rating: [53.16652],
    history_rating: [108.39066, 124.212105, 124.212105],
    good_noisy_margin: [-20.445045],
    good_noisy_bonus: [1000.0],
    aw_width: [4.9304986, 1.4202363, 1.4202363],
    aw_fl_lerp: [0.52635705],
    aw_fh_reduction: [1.0448276, 0.1, 3.0],
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

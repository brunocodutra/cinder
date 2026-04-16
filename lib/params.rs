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
    moves_left: [-1.0, 140.0],
    moves_left_limits: [1.0, 60.0],
    soft_time_fraction: [0.7399792],
    hard_time_fraction: [0.86297363],
    score_trend_inertia: [0.120955214],
    score_trend_scale: [1.1591196, 32.210957],
    pv_focus_scale: [-1.6761804, 1.8973947],
    piece_values: [52.847713, 188.47716, 215.22618, 306.7143, 629.2747, 0.0],
    material_scaling: [0.7, 1.05],
    halfmove_scaling: [1.0, 0.75],
    pawns_correction: [13.677705],
    minor_correction: [9.955085],
    major_correction: [10.968398],
    pieces_correction: [16.386993],
    continuation_correction: [12.0, 8.0],
    pawns_correction_delta: [0.00059015467, -0.25, 0.25],
    minor_correction_delta: [0.00057724636, -0.25, 0.25],
    major_correction_delta: [0.0006286239, -0.25, 0.25],
    pieces_correction_delta: [0.0006316802, -0.25, 0.25],
    counter_correction_delta: [0.0006, -0.25, 0.25],
    followup_correction_delta: [0.0006, -0.25, 0.25],
    attacker_history_bonus: [0.04, 0.003, 0.25],
    attacker_history_malus: [-0.04, -0.003, -0.25],
    defender_history_bonus: [0.04, 0.003, 0.25],
    defender_history_malus: [-0.04, -0.003, -0.25],
    butterfly_history_bonus: [0.04, 0.0028119748, 0.25],
    butterfly_history_malus: [-0.045, -0.006025375, -0.25],
    counter_history_bonus: [0.052, 0.0034933158, 0.25],
    counter_history_malus: [-0.042, -0.00282255, -0.25],
    followup_history_bonus: [0.052, 0.0034933158, 0.25],
    followup_history_malus: [-0.042, -0.00282255, -0.25],
    tt_cutoff_hm_limit: [87.07959],
    tb_depth_bonus: [3.8686926],
    nmr_depth_limit: [3.0],
    nmr_depth: [0.3055916],
    nmr_score: [0.1415365, 0.1190361, 3.447467],
    fhp_depth_limit: [2.0],
    fhp_margin_depth: [20.0, 20.0],
    fhp_margin_scalar: [200.0],
    razoring_depth: [8.4302845, 20.976559],
    razoring_scalar: [30.856945],
    rfp_margin_depth: [1.238703, 11.705757],
    rfp_margin_scalar: [9.479503],
    rfp_margin_is_improving: [-11.4111],
    probcut_depth: [0.009150021, -3.0125625],
    probcut_depth_bounds: [6.007756, 2.9932067],
    probcut_margin_depth: [30.0, 180.0],
    probcut_margin_is_improving: [-20.0],
    probcut_depth_bonus: [0.5048079],
    singular_depth: [0.38281223, -0.64462805],
    singular_depth_bounds: [6.037731, 3.0384016],
    singular_margin_depth: [0.48121896, 0.34409145],
    singular_extension_limit: [-3.0070076, 2.9203167],
    singular_extension_scalar: [1.1575831, 0.55394185, 0.56306934, 0.123578124],
    singular_extension_is_cut: [0.58390427, -1.0124613, 0.42781678],
    singular_extension_is_fh: [0.46751645, 0.5347338],
    singular_extension_is_quiet: [0.0328596],
    singular_extension_score: [1.2, 0.18],
    singular_reduction_scalar: [-0.049872175, -0.858315, -1.1109568],
    singular_reduction_is_cut: [-1.0591198, 1.078267],
    singular_reduction_is_fh: [-0.8221849],
    lmp_depth: [0.0, 0.8831169, 0.3232323],
    lmp_is_improving: [1.0584416, 1.0584416],
    lmp_scalar: [1.6767668],
    futility_depth_limit: [16.0],
    futility_margin_depth: [4.0, 12.0],
    futility_margin_scalar: [60.0],
    futility_margin_quiescence: [60.0],
    see_margin_noisy_depth: [-2.1050088, -47.118065],
    see_margin_noisy_scalar: [-20.0],
    see_margin_quiet_depth: [-8.862688, -0.8800826],
    see_margin_quiet_scalar: [-2.0],
    see_margin_quiescence: [-10.0],
    lmr_index: [0.01092667, 0.28172347, 0.04370668],
    lmr_depth: [0.04182772, 0.05906677],
    lmr_scalar: [0.39068964],
    lmr_is_root: [-0.26058814, -0.05927927, 0.014655725, -0.47737756, 0.064811245, -0.8506578],
    lmr_not_root: [-0.2773116, -0.3484747, 0.16766922, 0.14824785, 0.3444434, 1.0012585, 0.27381346, 0.20338072, -0.34402922, -0.12740706, -0.100679174, -0.055079766, -0.5635493, 0.55850875, -0.91804814, -1.132279],
    lmr_was_pv: [-0.40264055, 0.08336671, -0.04269617, -0.065438956, -0.069447584, -0.0134953465, 0.00027749967, -0.23074132, -0.018634617, -0.086658254, 0.0747456, -0.048453383, -0.019069955, -0.18566678, -0.020113625],
    lmr_was_all: [0.027016468, -0.011685938, 0.034589678, -0.034265377, -0.022528697, 0.042460904, 0.029170342, -0.08618233, 0.036730893, -0.06403259, -0.050650023, 0.11217485, -0.12831268, 0.100620456],
    lmr_is_all: [0.23545812, -0.05286877, -0.0039083054, 0.18338697, 0.05346083, 0.057920445, -0.050552703, -0.096128955, -0.20854609, 0.05567025, 0.08269207, -0.051368967, -0.10712282],
    lmr_was_cut: [0.3708072, 0.030704066, 0.0034351293, 0.013440257, 0.15817471, -0.7897408, -0.009228248, -0.006690263, 0.021954231, -0.09824374, -0.019543014, -0.009639917],
    lmr_is_cut: [0.97646433, 0.029022593, -0.029223155, -0.030863527, -0.023213226, -0.0035138943, -0.10073063, 0.13436058, -0.0185991, -0.05387889, -0.07281809],
    lmr_is_fl: [0.24007352, 0.0072098617, -0.11176458, 0.039058402, -0.111399174, 0.07298508, -0.043494835, 0.07188853, -0.022125915, 0.008343734],
    lmr_is_fh: [0.2360654, 0.07888371, 0.0021355308, 0.0740855, 0.07570441, -0.019655298, 0.026363604, 0.08935043, 0.14005981],
    lmr_is_improving: [-0.3417773, -0.023203216, -0.09727752, 0.100037456, -0.13064604, 0.021965388, -0.093397066, 0.027411625],
    lmr_was_quiet: [-0.00024217111, -0.013540098, 0.011007771, 0.07331288, -0.090631984, -0.13177897, -0.07402594],
    lmr_is_quiet: [-0.3497083, 0.06741455, -0.13838418, 0.0476304, -0.124961704, -0.041722402],
    lmr_is_check: [0.12550871, 0.06402838, 0.09017358, 0.0034720118, 0.03944079],
    lmr_gives_check: [-0.58770466, -0.065412626, 0.15423833, 0.08629131],
    lmr_raised_alpha: [0.12706119, -0.15894184, -0.15144041],
    lmr_butterfly: [0.12564152, 0.011221688],
    lmr_counter: [-0.1391397],
    lmr_extension: [-0.5788193],
    lmr_threshold: [0.45376918],
    move_rating_killer: [53.16652],
    move_rating_gives_check: [50.0],
    move_rating_history: [108.39066, 108.39066, 108.39066],
    move_rating_continuation: [0.0, 124.212105, 124.212105],
    move_rating_see: [10.0, 550.0, -10.0],
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

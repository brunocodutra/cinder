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
    iir_scalar: [-1.0811034, -0.17498612, -0.03123212],
    iir_is_all: [-0.07957493, 0.12252191],
    iir_is_cut: [-0.03123212],
    nmr_depth_limit: [3.0],
    nmr_depth: [0.3055916],
    nmr_score: [0.1415365, 0.1190361, 3.447467],
    fhp_depth_limit: [2.0],
    fhp_margin_scalar: [200.0, 20.0],
    fhp_margin_depth: [20.0],
    razoring_scalar: [30.856945, 20.976559],
    razoring_depth: [8.4302845],
    rfp_margin_scalar: [9.546398, 11.6784315, -5.7796087],
    rfp_margin_depth: [1.2450851, -0.11946355],
    rfp_margin_is_improving: [-5.895293],
    probcut_depth: [0.009150021, -3.0125625],
    probcut_depth_bounds: [6.007756, 2.9932067],
    probcut_margin_depth: [30.0, 180.0],
    probcut_margin_is_improving: [-20.0],
    probcut_depth_bonus: [0.5048079],
    singular_depth: [0.38256344, -0.8027754],
    singular_depth_bounds: [6.1580563, 2.893212],
    singular_margin_depth: [0.5710629, 0.37909773],
    singular_extension_limit: [-2.8605912, 2.9106972],
    singular_extension_scalar: [1.2843643, 0.53426945, 0.5047369, 0.11781449],
    singular_extension_is_cut: [0.87949955, -1.0095826, 0.4348788],
    singular_extension_is_fh: [0.5411726, 0.54582536],
    singular_extension_is_quiet: [0.14774026],
    singular_extension_score: [1.2, 0.18],
    singular_reduction_scalar: [0.18129678, -0.8734205, -0.9783077],
    singular_reduction_is_cut: [-1.0148416, 1.0842195],
    singular_reduction_is_fh: [-0.7698692],
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
    lmr_scalar: [0.2694574],
    lmr_is_root: [-0.24061991, -0.37810585, 0.113798216, -0.4888219, -0.014038441, -0.87258285],
    lmr_not_root: [-0.3474799, -0.60703963, 0.072568685, 0.19518234, 0.11469804, 1.0613369, 0.023536302, 0.025341254, -0.3736423, -0.03719023, -0.041707322, -0.023437526, -0.6606824, 0.7017346, -0.9793593, -1.085641],
    lmr_was_pv: [-0.5167015, -0.14382796, -0.030166112, -0.056280777, -0.06525844, -0.042439222, -0.22058941, -0.30427334, 0.009399751, -0.14281398, 0.042468537, 0.022812773, -0.14534207, -0.25294492, -0.07950489],
    lmr_was_all: [0.08618553, 0.0853578, -0.114813745, -0.11651498, 0.10587481, 0.09727, 0.11238848, -0.091358155, 0.0071076686, -0.23972756, -0.10631339, 0.23714145, -0.20680104, 0.0009631668],
    lmr_is_all: [0.37282792, -0.12094125, 0.18213893, 0.39784703, 0.13982509, -0.34258965, -0.11738683, -0.056642827, -0.13457575, 0.1149483, 0.22597256, -0.114281714, -0.16607569],
    lmr_was_cut: [0.5490275, 0.21706162, -0.10991277, -0.032485045, 0.081619784, -0.79652303, -0.16225983, -0.015630152, -0.028057564, -0.16298822, -0.11264752, 0.044111744],
    lmr_is_cut: [1.0835551, 0.15933432, -0.08533618, 0.048006695, -0.03891606, 0.17402942, 0.1059569, 0.33086333, 0.18555236, 0.057437662, -0.061297763],
    lmr_is_fl: [0.12905829, -0.1510492, -0.10480073, 0.34178135, -0.115631, -0.07093257, -0.128407, 0.08138201, -0.08108762, -0.16809328],
    lmr_is_fh: [0.4277691, -0.09194775, 0.074826, 0.061045744, 0.14141317, -0.18966024, 0.010230558, 0.031262103, 0.13313438],
    lmr_is_improving: [-0.31978264, -0.014250574, -0.06622811, 0.07050629, -0.23174594, 0.2094536, 0.014932545, -0.2004387],
    lmr_was_quiet: [0.05335545, 0.053355414, 0.018155843, 0.014747844, -0.040885955, -0.294088, -0.1479274],
    lmr_is_quiet: [-0.30707827, 0.24974062, -0.21631406, 0.22404568, -0.10625593, -0.14194393],
    lmr_is_check: [0.14102241, 0.14209257, 0.066650495, 0.16084816, 0.04467388],
    lmr_gives_check: [-0.6653356, -0.2204604, 0.20237486, 0.08065511],
    lmr_raised_alpha: [0.07909846, -0.0012781671, -0.005659628],
    lmr_butterfly: [0.057852622, 0.11749611],
    lmr_counter: [0.035936765],
    lmr_extension: [-0.6497572],
    lmr_threshold: [0.37115636],
    move_rating_is_killer: [53.16652],
    move_rating_gives_check: [50.0],
    move_rating_history: [108.39066, 108.39066, 108.39066],
    move_rating_continuation: [124.212105, 124.212105],
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

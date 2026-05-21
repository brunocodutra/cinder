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

impl<const BYTES: ConstBytes<64>> Default for Param<BYTES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const BYTES: ConstBytes<64>> Deref for Param<BYTES> {
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
            #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
            pub fn $name<R: SliceIndex<[f32]>>(idx: R) -> &'static R::Output {
                unsafe { PARAMS.get().as_ref_unchecked().$name.get_unchecked(idx) }
            }
        })*

        #[cfg(feature = "spsa")]
        impl Params {
            /// Total number of parameters.
            pub const LEN: usize = len!($($value,)*);

            /// Initializes the global params.
            pub const fn init(self) {
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
    moves_left: [-1.0368232, 138.06566],
    moves_left_limits: [0.9729303, 60.17618],
    soft_time_fraction: [0.71118027],
    hard_time_fraction: [0.811659],
    score_trend_inertia: [0.12196421],
    score_trend_scale: [1.0852572, 31.240652],
    pv_focus_scale: [-1.6350247, 2.0301223],
    piece_values: [50.594063, 187.51443, 215.74928, 306.34222, 604.27924, 0.0],
    material_scaling: [0.6830509, 1.1700959],
    halfmove_scaling: [1.1213975, 0.75001514],
    pawns_correction: [13.258422],
    minor_correction: [10.054037],
    major_correction: [11.305938],
    pieces_correction: [16.484673],
    history_correction: [12.412882, 11.96483],
    continuation_correction: [11.5984955, 8.063492],
    correction_gradient: [0.0006024718, -0.2321398, 0.25532514],
    history_bonus: [0.06047257, -0.019689497, 0.39126137],
    history_malus: [-0.059175313, 0.020342972, -0.3752189],
    tt_cutoff_hm_limit: [85.60747],
    tb_depth_bonus: [3.7609527],
    iir_scalar: [-1.1026508, -0.18468475, -0.031067677],
    iir_is_all: [-0.0802273, 0.12216754],
    iir_is_cut: [-0.029835153],
    nmr_depth_limit: [2.9495397],
    nmr_depth: [0.31896353],
    nmr_score: [0.13594417, 0.120037615, 3.4059312],
    fhp_depth_limit: [2.1012514],
    fhp_margin_scalar: [199.39035, 19.79673],
    fhp_margin_depth: [21.409462],
    razoring_scalar: [28.576336, 22.45721],
    razoring_depth: [8.531234],
    rfp_margin_scalar: [10.2519455, 11.809901, -5.821296],
    rfp_margin_depth: [1.2683023, -0.12537429],
    rfp_margin_is_improving: [-5.740662],
    probcut_depth: [0.00887401, -2.8750348],
    probcut_depth_bounds: [6.0162735, 3.0423841],
    probcut_margin_depth: [30.616125, 179.21928],
    probcut_margin_is_improving: [-20.852684],
    probcut_depth_bonus: [0.5366715],
    singular_depth: [0.40581933, -0.8278464],
    singular_depth_bounds: [6.72761, 3.058447],
    singular_margin_depth: [0.56260633, 0.38707882],
    singular_extension_limit: [-3.0552912, 2.9511032],
    singular_extension_scalar: [1.2762852, 0.54129696, 0.5210821, 0.11008555],
    singular_extension_is_cut: [0.81625485, -0.9768444, 0.43959242],
    singular_extension_is_fh: [0.51845807, 0.5212415],
    singular_extension_is_quiet: [0.15207285],
    singular_extension_score: [1.2918028, 0.18640476],
    singular_reduction_scalar: [0.17531087, -0.867881, -0.9855116],
    singular_reduction_is_cut: [-1.0740738, 1.0801995],
    singular_reduction_is_fh: [-0.7653252],
    lmp_depth: [0.0, 0.8529595, 0.3219452],
    lmp_is_improving: [1.0865138, 1.0330937],
    lmp_scalar: [1.7028192],
    futility_depth_limit: [16.425198],
    futility_margin_depth: [3.991961, 12.072424],
    futility_margin_scalar: [61.432102],
    futility_margin_quiescence: [59.018124],
    see_margin_noisy_depth: [-2.0773406, -44.936344],
    see_margin_noisy_scalar: [-19.69514],
    see_margin_quiet_depth: [-8.532175, -0.93644124],
    see_margin_quiet_scalar: [-2.0592804],
    see_margin_quiescence: [-10.133915],
    lmr_index: [0.011179409, 0.2827286, 0.044615123],
    lmr_depth: [0.043961063, 0.06588891],
    lmr_scalar: [0.28587267],
    lmr_is_root: [-0.24846461, -0.49418688, -0.3722782, 0.112787604, -0.44766632, -0.013118338],
    lmr_not_root: [-0.34969956, -0.58828676, 0.069692664, 0.1973518, 0.11463388, 1.1096525, 0.023956422, 0.024674295, -0.36725202, -0.03996993, -0.042235337, -0.023890581, -0.66345626, 0.68391913],
    lmr_was_pv: [-0.50736564, -0.13897552, -0.029762655, -0.06350709, -0.06636452, -0.040655475, -0.24577737, -0.27207184, 0.009023711, -0.14451124, 0.04297805, 0.02173699, -0.14534771],
    lmr_was_all: [0.08119077, 0.08611465, -0.11263096, -0.118116476, 0.10119027, 0.09456716, 0.11184974, -0.09696647, 0.0071593365, -0.21611467, -0.10870493, 0.23328882],
    lmr_is_all: [0.3847843, -0.12722652, 0.18419597, 0.39161855, 0.14039144, -0.3462014, -0.12064718, -0.057299722, -0.13697092, 0.115709625, 0.2356469],
    lmr_was_cut: [0.53815615, 0.2189696, -0.115462765, -0.031222789, 0.08196879, -0.815747, -0.15529609, -0.015866641, -0.026104284, -0.15391965],
    lmr_is_cut: [1.0610466, 0.15884896, -0.089040056, 0.046876237, -0.0405892, 0.17088127, 0.109668225, 0.356028, 0.19248573],
    lmr_is_fl: [0.132264, -0.15353253, -0.107403345, 0.32037386, -0.109378695, -0.07133506, -0.12815773, 0.08247008],
    lmr_is_fh: [0.397941, -0.09273756, 0.06982148, 0.061489005, 0.13192327, -0.19428265, 0.010192002],
    lmr_is_improving: [-0.31112778, -0.013840142, -0.057500947, 0.07541573, -0.21875395, 0.21188939],
    lmr_was_quiet: [0.054972872, 0.051171765, 0.016991349, 0.0143032875, -0.040167022],
    lmr_is_quiet: [-0.32864192, 0.24548467, -0.23078659, 0.23957407],
    lmr_is_check: [0.14302789, 0.1493448, 0.061481126],
    lmr_gives_check: [-0.70523965, -0.23624316],
    lmr_raised_alpha: [0.07508132],
    lmr_history: [-1.0183239],
    lmr_continuation: [-1.0939374],
    lmr_extension: [-0.6577877],
    lmr_threshold: [0.3429049],
    move_rating_is_killer: [52.501087],
    move_rating_gives_check: [52.60514],
    move_rating_history: [110.318756, 101.72177, 110.25604],
    move_rating_continuation: [120.90195, 124.31798, 0.0, 39.09851],
    move_rating_see: [10.015516, 508.8259, -10.646439],
    aw_width: [4.7896934, 1.3622901, 1.3989353],
    aw_fl_lerp: [0.5291054],
    aw_fh_reduction: [1.0742917, 0.09090758, 3.048758],
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

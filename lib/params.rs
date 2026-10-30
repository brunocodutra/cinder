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
    moves_left: [0.0076926323, 0.038901687],
    time_limits: [0.56793386, 0.81476027],
    score_trend_inertia: [0.13477112],
    score_trend_scale: [1.1449345, 31.25194],
    pv_focus_scale: [-1.4857069, 1.920175],
    piece_values: [51.76566, 192.70207, 202.70927, 353.73688, 655.53925, 0.0],
    piece_values_scaling: [52.282032, 185.24663, 205.27357, 361.68713, 641.96857, 0.0],
    material_scaling: [0.69900084, 1.0344105],
    halfmove_scaling: [1.2889253, 0.7367995],
    fail_firm_pivot: [1.0428573],
    pawns_correction: [16.049494],
    minor_correction: [9.293415],
    major_correction: [11.701546],
    pieces_correction: [16.800493],
    history_correction: [12.459983, 13.585891],
    continuation_correction: [11.196399, 7.504085],
    correction_gradient: [0.0005516094, -0.21156867, 0.26831418],
    history_bonus: [0.04960208, -0.022677926, 0.39788538],
    history_malus: [-0.06484208, 0.019743338, -0.31086174],
    tt_cutoff_hm_limit: [88.03843],
    tb_depth_bonus: [3.7417564],
    iir_scalar: [-1.2102877, -0.16880709, -0.032057256],
    iir_is_all: [-0.08760728, 0.10239702],
    iir_is_cut: [-0.028823908],
    nmr_depth_limit: [2.7057242],
    nmr_margin_scalar: [8.5209, -0.2069, -0.6735],
    nmr_margin_depth: [-0.000625, 0.025],
    nmr_margin_is_improving: [-0.6735],
    nmr_depth_scalar: [0.13366544, 0.14075977, 0.005],
    nmr_depth_margin: [0.0001, 0.0001],
    nmr_depth_is_improving: [0.005],
    nmr_depth: [0.6283978],
    nmr_limit: [3.1927316],
    nmv_depth_limit: [16.0],
    nmv_draft: [0.75, -0.75],
    fhp_depth_limit: [2.2888892],
    fhp_margin_scalar: [157.47755, 18.604353],
    fhp_margin_depth: [19.562542],
    quiescence_sp_lerp: [0.9],
    quiescence_fh_lerp: [0.9],
    razoring_depth_limit: [3.9867766],
    razoring_scalar: [31.123924, 22.94282],
    razoring_depth: [7.881912],
    rfp_depth_limit: [11.987534],
    rfp_margin_scalar: [9.799021, 10.478613, -6.4054923],
    rfp_margin_depth: [1.3605858, -0.13562155],
    rfp_margin_is_improving: [-6.518849],
    rfp_lerp: [0.8],
    probcut_depth: [0.008522523, -3.365653],
    probcut_depth_bounds: [5.1959376, 2.8179011],
    probcut_margin_depth: [30.35216, 155.76288],
    probcut_margin_is_improving: [-22.31931],
    probcut_depth_bonus: [0.5247597],
    sext_depth: [0.46864784, -0.92318904],
    sext_depth_bounds: [6.902374, 2.7317097],
    sext_margin_depth: [0.5819546, 0.41399035],
    sext_dominance_model: [-0.11538522, 0.14604612],
    sext_scalar: [0.1715122, -0.0035679415, -1.0300871, -0.90388316, 0.25694117, 0.33051008, 0.2918851, 0.26114494],
    sext_is_quiet: [-0.0039923275, 0.0, 0.0, 0.23075992, -0.16186912, -0.16029952, 0.031177115],
    sext_is_cut: [-1.0205399, 1.0555462, 0.34864467, 0.3608637, -0.46070123, 0.0],
    sext_is_fh: [-0.84649605, 0.28167775, -0.45580506, 0.29369858, 0.0],
    sext_is_singular: [0.23688765, 0.35781986, 0.28659067, 1.0109847],
    sext_is_singular_cut: [0.35684198, -0.46207833, 1.2006514],
    sext_is_singular_fh: [0.26747355, 1.1467329],
    sext_dominance: [-0.5625563],
    lmp_scalar: [1.6376399, 0.26472557, 0.98742265],
    lmp_depth: [0.0, 0.76364875],
    lmp_is_improving: [1.0014535],
    futility_depth_limit: [16.07704],
    futility_margin_quiescence: [56.39047],
    futility_margin_scalar: [59.174576, 11.180963],
    futility_margin_depth: [3.5208318],
    see_margin_noisy_depth: [-2.1658046, -53.154556],
    see_margin_noisy_scalar: [-20.319162],
    see_margin_quiet_depth: [-8.270519, -0.9482466],
    see_margin_quiet_scalar: [-2.2544768],
    see_margin_quiescence: [-11.59767],
    lmr_scalar: [0.24907432, 0.046417363, 0.061074838],
    lmr_index: [0.010666319, 0.29299894],
    lmr_depth: [0.04006693],
    lmr_is_root: [-0.24448578, -0.49192807, -0.3655567, 0.045141406, -0.43116447, 0.0036023809],
    lmr_not_root: [-0.4008434, -0.5773772, 0.062898904, 0.19030909, 0.14775996, 1.0775461, 0.010656108, -0.0044799107, -0.33210444, -0.08880723, -0.028926019, -0.048448376, -0.65708524, 0.59380776],
    lmr_was_pv: [-0.5123212, -0.15551437, -0.018824667, -0.05319458, -0.08653981, 0.017366061, -0.24289724, -0.28404546, 0.028774716, -0.1611347, 0.048943225, 0.025008455, -0.20811394],
    lmr_was_all: [0.0635758, 0.047968093, -0.14753132, -0.12361287, 0.11104315, 0.0554482, 0.13825157, -0.09841054, -0.008804401, -0.20745361, -0.10091524, 0.23915862],
    lmr_is_all: [0.41152605, -0.18496946, 0.17499185, 0.40831336, 0.10596047, -0.33713007, -0.13166882, -0.026420828, -0.13107403, 0.13378447, 0.2223978],
    lmr_was_cut: [0.5393799, 0.26456374, -0.11743612, -0.044784933, 0.06278028, -0.7920262, -0.11249478, -0.037586737, 0.00040844938, -0.16392933],
    lmr_is_cut: [1.1552838, 0.13150175, -0.07468449, 0.09279424, -0.019651866, 0.15507536, 0.12043032, 0.33197924, 0.23989508],
    lmr_is_fl: [0.11560304, -0.12585965, -0.07112941, 0.37922645, -0.110646926, -0.06893509, -0.106586814, 0.08510499],
    lmr_is_fh: [0.43472904, -0.13687459, 0.13233626, 0.055728316, 0.14109333, -0.19427194, 0.01478708],
    lmr_is_improving: [-0.27999866, 0.00806233, -0.07450902, 0.05192276, -0.18336098, 0.2225794],
    lmr_was_quiet: [0.043718662, -0.024595564, 0.03203509, 0.026430141, -0.02578544],
    lmr_is_quiet: [-0.34357354, 0.24099563, -0.24395007, 0.2308002],
    lmr_is_check: [0.15503839, 0.14237787, 0.06726729],
    lmr_gives_check: [-0.59647995, -0.2788818],
    lmr_raised_alpha: [0.05579673],
    lmr_history: [-0.87951344],
    lmr_continuation: [-1.0829387],
    lmr_extension: [-0.7596345],
    lmr_threshold: [0.38200438],
    move_rating_is_killer: [51.01637],
    move_rating_gives_check: [57.107807],
    move_rating_history: [107.45497, 108.06068, 108.0759],
    move_rating_continuation: [130.66805, 117.88957, 0.0, 40.538338],
    move_rating_see: [9.46816, 449.19186, -12.358575],
    aw_width: [4.6744747, 1.1388648, 1.3412765],
    aw_fl_lerp: [0.5361622],
    aw_fh_reduction: [1.0469723, 0.09269802, 2.8279958],
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

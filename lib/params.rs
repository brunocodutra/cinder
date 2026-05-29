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
    moves_left: [-1.0755961, 143.70827],
    moves_left_limits: [0.96458066, 62.72076],
    soft_time_fraction: [0.65585905],
    hard_time_fraction: [0.80760103],
    score_trend_inertia: [0.12432647],
    score_trend_scale: [1.1391279, 30.397293],
    pv_focus_scale: [-1.604002, 2.0323455],
    piece_values: [48.814316, 186.3668, 201.55121, 318.24698, 665.5232, 0.0],
    material_scaling: [0.67822653, 1.1309454],
    halfmove_scaling: [1.2082757, 0.7544682],
    pawns_correction: [13.491723],
    minor_correction: [9.950712],
    major_correction: [12.0736],
    pieces_correction: [16.381235],
    history_correction: [12.478539, 12.103803],
    continuation_correction: [11.411582, 8.153436],
    correction_gradient: [0.00061456114, -0.23754033, 0.25952125],
    history_bonus: [0.059110697, -0.020378703, 0.40956506],
    history_malus: [-0.06159502, 0.020482974, -0.35126263],
    tt_cutoff_hm_limit: [81.0118],
    tb_depth_bonus: [3.778563],
    iir_scalar: [-1.1382314, -0.17286427, -0.032511137],
    iir_is_all: [-0.080907375, 0.11107357],
    iir_is_cut: [-0.030346677],
    nmr_depth_limit: [2.7164233],
    nmr_depth: [0.35717085],
    nmr_score: [0.13390744, 0.12326953, 3.4197507],
    fhp_depth_limit: [2.0730035],
    fhp_margin_scalar: [206.28088, 19.10656],
    fhp_margin_depth: [21.261944],
    razoring_scalar: [30.394001, 21.903364],
    razoring_depth: [8.20692],
    rfp_margin_scalar: [10.167637, 11.695678, -6.098894],
    rfp_margin_depth: [1.2796, -0.1273198],
    rfp_margin_is_improving: [-6.0708036],
    probcut_depth: [0.00875227, -3.1612828],
    probcut_depth_bounds: [5.6557636, 3.139031],
    probcut_margin_depth: [30.528044, 161.6239],
    probcut_margin_is_improving: [-21.605963],
    probcut_depth_bonus: [0.5352057],
    singular_depth: [0.44131786, -0.8387809],
    singular_depth_bounds: [6.596972, 2.9919279],
    singular_margin_depth: [0.56598276, 0.38505372],
    singular_extension_limit: [-3.0760343, 2.740192],
    singular_extension_scalar: [1.2303861, 0.52791774, 0.5361313, 0.10537616],
    singular_extension_is_cut: [0.81403685, -0.9450906, 0.42739695],
    singular_extension_is_fh: [0.5138978, 0.46994835],
    singular_extension_is_quiet: [0.15215072],
    singular_extension_score: [1.2834897, 0.17760244],
    singular_reduction_scalar: [0.17082599, -0.98146635, -1.0004929],
    singular_reduction_is_cut: [-1.0985963, 1.0326279],
    singular_reduction_is_fh: [-0.7520062],
    lmp_depth: [0.0, 0.8388554, 0.31931564],
    lmp_is_improving: [1.0066552, 1.0630146],
    lmp_scalar: [1.6895659],
    futility_depth_limit: [15.315033],
    futility_margin_depth: [3.9734087, 11.080011],
    futility_margin_scalar: [64.64284],
    futility_margin_quiescence: [57.66092],
    see_margin_noisy_depth: [-1.9475416, -47.76224],
    see_margin_noisy_scalar: [-19.786987],
    see_margin_quiet_depth: [-8.093317, -0.91751593],
    see_margin_quiet_scalar: [-2.140722],
    see_margin_quiescence: [-10.193605],
    lmr_index: [0.0110635245, 0.27730855, 0.046392232],
    lmr_depth: [0.042156953, 0.066948116],
    lmr_scalar: [0.30253264],
    lmr_is_root: [-0.25952628, -0.49478638, -0.39170906, 0.107451566, -0.46631932, -0.011773334],
    lmr_not_root: [-0.36044633, -0.60743487, 0.06606396, 0.19359577, 0.11377966, 1.1243681, 0.023268618, 0.027019767, -0.37309182, -0.036570303, -0.046537567, -0.025029141, -0.6787738, 0.6649634],
    lmr_was_pv: [-0.5275338, -0.14116323, -0.028162677, -0.064810336, -0.061631247, -0.045067616, -0.24978419, -0.26436684, 0.0091381315, -0.14379105, 0.041158613, 0.022987148, -0.13909741],
    lmr_was_all: [0.0830754, 0.081486315, -0.12116738, -0.12615332, 0.09720646, 0.097170055, 0.11610359, -0.092532456, 0.007133969, -0.21496865, -0.11075207, 0.2270784],
    lmr_is_all: [0.38630202, -0.13064867, 0.17288126, 0.39863318, 0.1372157, -0.3719237, -0.122507095, -0.058746077, -0.12957102, 0.11655734, 0.24346405],
    lmr_was_cut: [0.51733094, 0.22912133, -0.121064395, -0.03134368, 0.08430214, -0.762213, -0.15791209, -0.015923325, -0.025983382, -0.16680644],
    lmr_is_cut: [1.1488515, 0.15849502, -0.096312575, 0.04886621, -0.04346599, 0.16049072, 0.108023666, 0.34894758, 0.20268966],
    lmr_is_fl: [0.12744148, -0.1518919, -0.11423856, 0.36124712, -0.1131816, -0.07142896, -0.12034576, 0.08172089],
    lmr_is_fh: [0.400876, -0.09180612, 0.06765787, 0.05592212, 0.12054242, -0.2129631, 0.0099946065],
    lmr_is_improving: [-0.29940146, -0.01447181, -0.06057251, 0.073212415, -0.22270648, 0.20739299],
    lmr_was_quiet: [0.05589651, 0.048614237, 0.016817182, 0.01461129, -0.040728763],
    lmr_is_quiet: [-0.32283673, 0.25395468, -0.24146076, 0.25592366],
    lmr_is_check: [0.14609507, 0.14347583, 0.061033044],
    lmr_gives_check: [-0.68147373, -0.23885413],
    lmr_raised_alpha: [0.08120262],
    lmr_history: [-0.9649463],
    lmr_continuation: [-1.1244438],
    lmr_extension: [-0.6432393],
    lmr_threshold: [0.36594504],
    move_rating_is_killer: [53.94221],
    move_rating_gives_check: [56.538944],
    move_rating_history: [109.17189, 102.351425, 106.94179],
    move_rating_continuation: [125.81573, 119.723366, 0.0, 38.29103],
    move_rating_see: [9.636155, 444.4059, -11.599353],
    aw_width: [4.443728, 1.2247118, 1.3201699],
    aw_fl_lerp: [0.53154814],
    aw_fh_reduction: [1.0501214, 0.084733576, 2.9634538],
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

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
    moves_left: [-1.1250302, 129.7077],
    moves_left_limits: [0.9204693, 55.388912],
    soft_time_fraction: [0.58262795],
    hard_time_fraction: [0.81017804],
    score_trend_inertia: [0.12942228],
    score_trend_scale: [1.0891559, 30.186228],
    pv_focus_scale: [-1.5344104, 1.9312489],
    piece_values: [50.39684, 184.75441, 201.29071, 349.0056, 632.258, 0.0],
    material_scaling: [0.67067015, 1.0132211],
    halfmove_scaling: [1.2637877, 0.7495756],
    pawns_correction: [15.2837515],
    minor_correction: [9.531928],
    major_correction: [11.519005],
    pieces_correction: [16.11618],
    history_correction: [12.55552, 13.257767],
    continuation_correction: [10.961311, 7.8432827],
    correction_gradient: [0.00059688155, -0.21099786, 0.26772803],
    history_bonus: [0.051497757, -0.023285924, 0.3924306],
    history_malus: [-0.062780336, 0.02069643, -0.3066847],
    tt_cutoff_hm_limit: [86.08928],
    tb_depth_bonus: [3.705568],
    iir_scalar: [-1.1548024, -0.16876285, -0.03241522],
    iir_is_all: [-0.08647807, 0.10061995],
    iir_is_cut: [-0.028648594],
    nmr_depth_limit: [2.6817758],
    nmr_depth: [0.3716022],
    nmr_score: [0.14075977, 0.13366544, 3.1927316],
    fhp_depth_limit: [2.2592714],
    fhp_margin_scalar: [170.64804, 18.890638],
    fhp_margin_depth: [19.776033],
    razoring_scalar: [30.0234, 21.85148],
    razoring_depth: [8.186248],
    rfp_margin_scalar: [10.407456, 10.813848, -6.300178],
    rfp_margin_depth: [1.3744049, -0.13397427],
    rfp_margin_is_improving: [-6.637267],
    probcut_depth: [0.008366786, -3.3176877],
    probcut_depth_bounds: [5.534753, 2.9517484],
    probcut_margin_depth: [29.81105, 161.51877],
    probcut_margin_is_improving: [-22.479607],
    probcut_depth_bonus: [0.5215299],
    singular_depth: [0.4648013, -0.8973641],
    singular_depth_bounds: [7.0152187, 2.7650673],
    singular_margin_depth: [0.59010494, 0.41410103],
    singular_extension_limit: [-3.025499, 2.6753526],
    singular_extension_scalar: [1.1590734, 0.47219053, 0.57133687, 0.1685826],
    singular_extension_is_cut: [0.81675416, -0.8895057, 0.4370304],
    singular_extension_is_fh: [0.49158373, 0.45050246],
    singular_extension_is_quiet: [0.16730642],
    singular_extension_score: [1.3185018, 0.18964037],
    singular_reduction_scalar: [0.19168185, -1.0052328, -0.9192594],
    singular_reduction_is_cut: [-1.0925053, 1.0689297],
    singular_reduction_is_fh: [-0.7917767],
    lmp_depth: [0.0, 0.73554915, 0.26149723],
    lmp_is_improving: [0.9943649, 1.0494918],
    lmp_scalar: [1.5982815],
    futility_depth_limit: [16.538027],
    futility_margin_depth: [3.6760995, 11.43988],
    futility_margin_scalar: [60.54757],
    futility_margin_quiescence: [56.233845],
    see_margin_noisy_depth: [-2.1658046, -53.154556],
    see_margin_noisy_scalar: [-20.319162],
    see_margin_quiet_depth: [-8.270519, -0.9482466],
    see_margin_quiet_scalar: [-2.2544768],
    see_margin_quiescence: [-11.668562],
    lmr_index: [0.010491702, 0.30122906, 0.04474858],
    lmr_depth: [0.039283242, 0.0636802],
    lmr_scalar: [0.25658995],
    lmr_is_root: [-0.24832498, -0.5047384, -0.38421553, 0.046979357, -0.41633716, 0.003548261],
    lmr_not_root: [-0.41001517, -0.57193, 0.0633806, 0.19040738, 0.14286122, 1.0277214, 0.010663405, -0.0043948847, -0.3358588, -0.088001855, -0.029481856, -0.0465716, -0.6450417, 0.61041886],
    lmr_was_pv: [-0.5074266, -0.15506758, -0.018273413, -0.053175483, -0.089505024, 0.016994758, -0.23813544, -0.2788035, 0.029828835, -0.1644435, 0.04972254, 0.025011789, -0.21617953],
    lmr_was_all: [0.06037189, 0.04736305, -0.14805265, -0.13314529, 0.108222075, 0.058660872, 0.13827427, -0.09462869, -0.008384754, -0.19936976, -0.098690204, 0.23477517],
    lmr_is_all: [0.4399781, -0.17998368, 0.17601185, 0.4074895, 0.109005384, -0.3427179, -0.13007727, -0.02663156, -0.12930557, 0.12962317, 0.21478935],
    lmr_was_cut: [0.5228288, 0.2721043, -0.12086374, -0.046285413, 0.065587305, -0.775365, -0.116556615, -0.03765333, 0.00040474813, -0.16769731],
    lmr_is_cut: [1.2074212, 0.14328216, -0.081664115, 0.095675945, -0.020064496, 0.15151744, 0.12335975, 0.33007824, 0.24220142],
    lmr_is_fl: [0.11569932, -0.1351063, -0.0741559, 0.36294073, -0.11259583, -0.072286665, -0.10320736, 0.08243519],
    lmr_is_fh: [0.43224135, -0.13680404, 0.12972717, 0.05738645, 0.14193368, -0.1986237, 0.015316214],
    lmr_is_improving: [-0.2723463, 0.007934878, -0.08174246, 0.052291047, -0.1910086, 0.21721332],
    lmr_was_quiet: [0.043741662, -0.025132371, 0.031345807, 0.027092967, -0.026773121],
    lmr_is_quiet: [-0.3429446, 0.23543467, -0.25238094, 0.220608],
    lmr_is_check: [0.15334196, 0.1400551, 0.06699711],
    lmr_gives_check: [-0.60851115, -0.2744671],
    lmr_raised_alpha: [0.0569849],
    lmr_history: [-0.8550789],
    lmr_continuation: [-1.0913005],
    lmr_extension: [-0.74210477],
    lmr_threshold: [0.3723231],
    move_rating_is_killer: [54.31386],
    move_rating_gives_check: [60.148857],
    move_rating_history: [106.14695, 103.28453, 106.111694],
    move_rating_continuation: [131.14809, 124.9024, 0.0, 40.846813],
    move_rating_see: [9.241001, 443.04495, -12.084356],
    aw_width: [4.533538, 1.0938935, 1.287042],
    aw_fl_lerp: [0.53142595],
    aw_fh_reduction: [1.0883383, 0.08622353, 2.889818],
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

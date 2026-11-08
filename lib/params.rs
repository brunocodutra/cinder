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
    nmr_margin_improving_model: [2.5, -2.5],
    nmr_margin_scalar: [8.520685, -0.206682, -2.635095],
    nmr_margin_depth: [-0.00063, 0.02531352],
    nmr_margin_improving: [1.2840767],
    nmr_depth_improving_model: [0.33333334, -1.6666666],
    nmr_depth_scalar: [0.13365545, 0.14075966, 0.02921619],
    nmr_depth_margin: [0.0001, 0.00010070826],
    nmr_depth_improving: [-0.019252222],
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
    rfp_improving_model: [4.0, -4.4],
    rfp_margin_scalar: [9.809828, 10.478865, -44.73562],
    rfp_margin_depth: [1.360606, -0.13629495],
    rfp_margin_improving: [31.80336],
    rfp_lerp: [0.8],
    probcut_depth: [0.008522523, -3.365653],
    probcut_depth_bounds: [5.1959376, 2.8179011],
    probcut_improving_model: [5.0, -3.75],
    probcut_margin_depth: [30.354625, 155.81411],
    probcut_margin_improving: [-22.511196],
    probcut_depth_bonus: [0.5247597],
    sext_depth: [0.46864784, -0.92318904],
    sext_depth_bounds: [6.902374, 2.7317097],
    sext_margin_depth: [0.5819546, 0.41399035],
    sext_dominance_model: [0.11538522, -0.14604612],
    sext_scalar: [0.1715122, -0.0035679415, -1.0300871, -0.90388316, 0.25694117, 0.33051008, 0.2918851, 0.26114494],
    sext_is_quiet: [-0.0039923275, 0.0, 0.0, 0.23075992, -0.16186912, -0.16029952, 0.031177115],
    sext_is_cut: [-1.0205399, 1.0555462, 0.34864467, 0.3608637, -0.46070123, 0.0],
    sext_is_fh: [-0.84649605, 0.28167775, -0.45580506, 0.29369858, 0.0],
    sext_is_singular: [0.23688765, 0.35781986, 0.28659067, 1.0109847],
    sext_is_singular_cut: [0.35684198, -0.46207833, 1.2006514],
    sext_is_singular_fh: [0.26747355, 1.1467329],
    sext_dominance: [-0.5625563],
    lmp_improving_model: [13.333333, -6.6666665],
    lmp_scalar: [1.6376546, 0.2647244, -3.4378998],
    lmp_depth: [0.0, 0.7636685],
    lmp_improving: [5.427683],
    futility_depth_limit: [16.07704],
    futility_margin_quiescence: [56.39047],
    futility_margin_scalar: [59.174576, 11.180963],
    futility_margin_depth: [3.5208318],
    see_margin_scalar: [-20.319162, -53.154556, 9.032343, 17.402103],
    see_margin_depth: [-2.1658046, 17.402103, -3.0523572],
    see_margin_is_quiet: [9.032343, 17.402103],
    see_margin_depth_is_quiet: [-3.0523572],
    see_margin_quiescence: [-11.59767],
    lmr_raising_alpha_model: [10.0, 4.5],
    lmr_improving_model: [6.6666665, -4.6666665],
    lmr_scalar: [0.24907432, 0.046417363, 0.061074838],
    lmr_index: [0.010666319, 0.29299894],
    lmr_depth: [0.04006693],
    lmr_is_root: [-0.4012366, -0.53964245, -0.21104752, 0.045141406, -0.38066107, 0.05],
    lmr_not_root: [-0.595233, -0.69297177, 0.05, 0.35142297, 0.12508424, 0.99564964, 0.010656108, -0.0044799107, -1.5014937, -0.05, -0.028926019, -0.048448376, -0.591207, 0.7998489],
    lmr_was_pv: [-0.54609954, -0.13634965, -0.018824667, -0.05, -0.055417378, 0.017366061, -0.24411646, -0.2984827, 0.028774716, -0.115750775, 0.048943225, 0.025008455, -0.20758775],
    lmr_was_all: [0.05, 0.047968093, -0.1831333, -0.11625737, 0.11308765, 0.0804482, 0.16667168, -0.17967962, -0.008804401, -0.052729737, -0.06907139, 0.2298849],
    lmr_is_all: [0.3234232, -0.10633609, 0.18749593, 0.40177006, 0.05, -0.37999433, -0.08367392, -0.026420828, -0.17951982, 0.16706137, 0.17248887],
    lmr_was_cut: [0.56581306, 0.2264697, -0.15871806, -0.044784933, 0.090064265, -0.8715615, -0.1729315, -0.037586737, 0.00040844938, -0.15678549],
    lmr_is_cut: [1.4725782, 0.05, -0.12028674, 0.05, -0.019651866, 0.15097667, 0.26649135, 0.48292336, 0.13461941],
    lmr_is_fl: [0.19575308, -0.16292982, -0.06300139, 0.38143831, -0.16926846, -0.05, -0.07112411, 0.05],
    lmr_is_fh: [0.4535835, -0.12337123, 0.1342868, 0.055062912, 0.24769594, -0.16112038, 0.01478708],
    lmr_improving: [0.76321644, 0.008097658, -0.07022782, 0.73769754, -0.18337382, 0.27633148],
    lmr_was_quiet: [0.043718662, -0.024595564, 0.03203509, 0.026430141, -0.02578544],
    lmr_is_quiet: [-0.4831159, 0.06415632, -0.1928208, 0.1564181],
    lmr_is_check: [0.05, 0.22218528, 0.29110688],
    lmr_gives_check: [-0.8715297, -0.2733629],
    lmr_raising_alpha: [-0.07398207],
    lmr_history_scalar: [0.1, -0.5265972, -0.5082307, -0.5451921, -1.0326369, -0.6193348],
    lmr_history_attacker: [0.1, -0.1, -0.1, -0.1, -0.1],
    lmr_history_defender: [0.1, -0.1, -0.1, -0.1],
    lmr_history_butterfly: [0.1, -0.1, -0.1],
    lmr_history_counter: [0.27084026, 0.1],
    lmr_history_followup: [0.1],
    lmr_extension: [-0.7596345],
    lmr_threshold: [0.38200438],
    move_rating_is_killer: [51.01637],
    move_rating_gives_check: [57.107807],
    move_rating_scalar: [0.0, 107.72594, 107.59123, 108.289313],
    move_rating_defender: [-1.0, 1.089035, -1.0],
    move_rating_attacker: [-1.0, 1.0],
    move_rating_butterfly: [1.0],
    move_rating_continuation_scalar: [0.0, 130.70897, 117.91428, 40.52512],
    move_rating_continuation1: [-1.0, -1.0, 1.0],
    move_rating_continuation2: [1.0, -1.0],
    move_rating_continuation4: [1.0],
    move_rating_see: [1.0, 332.1789, -47.442364],
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

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
    material_scaling: [0.70, 1.05],
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
    singular_depth: [0.46166527, -0.45030755],
    singular_depth_bounds: [6.0489254, 3.0092535],
    singular_margin_depth: [0.37896824, 0.38492644],
    singular_extension_limit: [-3.0, 3.0],
    singular_extension_scalar: [0.9590033, 0.5166677, 0.49674225, 0.04633785],
    singular_extension_is_cut: [0.55990803, -1.0195022, 0.49261725],
    singular_extension_is_fh: [0.50258714, 0.5289707],
    singular_extension_is_quiet: [0.00595205],
    singular_extension_score: [1.2, 0.18],
    singular_reduction_scalar: [-0.07166986, -1.0050749, -0.9766325],
    singular_reduction_is_cut: [-1.0225815, 1.0216024],
    singular_reduction_is_fh: [-0.9975375],
    lmp_depth: [0.0, 0.8831169, 0.3232323],
    lmp_is_improving: [1.0584416, 1.0584416],
    lmp_scalar: [1.6767667],
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
    lmr_scalar: [0.51642954],
    lmr_is_root: [-0.24884121, -0.12260032, -0.45845088, 0.25, -0.88297653],
    lmr_not_root: [-0.3199292, -0.16688395, 0.25, 0.9045571, 0.15648863, 0.15257305, -0.38831563, 0.36093542, -0.12260032, -0.45248434, 0.75, -0.8882448, -1.2104648],
    lmr_was_pv: [-0.18071137, 0.0464497, 0.022662345, 0.0, -0.0001325397, -0.046965536, 0.00601395, 0.014041026, 0.05279264, 0.0, -0.0046741012, -0.004452182],
    lmr_is_all: [0.25, 0.0, 0.25, -0.0034177457, 0.038919132, 0.017962204, -0.0014320713, 0.018522063, 0.0, -0.05525122, -0.0024242094],
    lmr_is_cut: [1.0199181, 0.0, 0.038298664, 0.002856032, 0.02266238, -0.0046429626, 0.0567602, 0.0, -0.05279855, -0.03311131],
    lmr_is_fl: [0.15648863, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    lmr_is_fh: [0.15990638, -0.027859548, -0.012490125, 0.023248037, 0.00036535802, 0.0, 0.019958453, -0.009807673],
    lmr_is_improving: [-0.42723476, -0.00880438, -0.031374026, -0.03538572, -0.01, -0.0039049838, -0.00018328608],
    lmr_is_noisy_node: [0.3222924, 0.010962273, 0.021270523, 0.0, 0.012177778, 0.01531091],
    lmr_is_quiet: [-0.16071045, -0.025325937, 0.0, 0.0047326805, 0.028538695],
    lmr_gives_check: [-0.46777619, 0.0, 0.016602615, 0.010076814],
    lmr_alpha_raises: [0.25, 0.0, 0.0],
    lmr_butterfly: [0.033187184, 0.039511792],
    lmr_counter: [0.033745848],
    lmr_extension: [-0.46644744],
    lmr_threshold: [0.48005262],
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

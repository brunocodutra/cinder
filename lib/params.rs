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
    #[cfg_attr(test, strategy(vec(-1e3f32..=1e3f32, Self::VALUES.len().cast::<usize>()).prop_map(Seq::from_iter)))]
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
    pawn_values: [39.83646, 53.672882, 51.00828, 57.46445, 60.383827, 55.03525, 55.17393, 50.206635],
    knight_values: [239.07317, 180.86995, 197.14519, 188.1538, 189.38432, 177.1756, 170.63026, 165.38493],
    bishop_values: [245.41153, 208.63464, 211.6043, 222.13768, 218.3559, 211.42186, 204.8089, 199.43465],
    rook_values: [245.34126, 319.40695, 329.34912, 324.7902, 311.54745, 323.54272, 306.86047, 292.87622],
    queen_values: [551.4669, 605.93115, 674.5755, 667.47955, 639.4869, 646.31274, 608.58704, 640.358],
    pawns_correction: [13.677705],
    minor_correction: [9.955085],
    major_correction: [10.968398],
    pieces_correction: [16.386993],
    pawns_correction_bonus: [1.0071973],
    minor_correction_bonus: [0.9851671],
    major_correction_bonus: [1.0728514],
    pieces_correction_bonus: [1.0780675],
    history_bonus_depth: [0.0, 5.133383],
    history_bonus_scalar: [0.71986556],
    continuation_bonus_depth: [0.0, 6.5756135],
    continuation_bonus_scalar: [0.89428884],
    history_penalty_depth: [0.0, -5.796621],
    history_penalty_scalar: [-1.542496],
    continuation_penalty_depth: [0.0, -5.5011106],
    continuation_penalty_scalar: [-0.7225728],
    probcut_margin_depth: [0.0, 13.350125],
    probcut_margin_scalar: [209.42744],
    single_extension_margin_depth: [0.0, 0.690351],
    single_extension_margin_scalar: [0.546739],
    double_extension_margin_depth: [0.0, 1.0699644],
    double_extension_margin_scalar: [0.26140416],
    triple_extension_margin_depth: [0.0, 0.5904154],
    triple_extension_margin_scalar: [144.23958],
    tt_cut_halfmove_limit: [87.07959],
    tb_cut_depth_bonus: [3.8686926],
    flp_margin_depth: [24.092096, 343.5552],
    flp_margin_scalar: [-132.51001],
    fhp_margin_depth: [32.207886, 63.049282],
    fhp_margin_scalar: [-47.862656],
    nmp_margin_depth: [0.55535954, 10.07757],
    nmp_margin_scalar: [-4.991574],
    nmr_gamma: [0.13003238],
    nmr_delta: [0.11947278],
    nmr_limit: [3.219951],
    nmr_fraction: [0.26338834],
    razoring_depth: [8.4302845, 20.976559],
    razoring_scalar: [30.856945],
    rfp_margin_depth: [1.238703, 11.705757],
    rfp_margin_scalar: [9.479503],
    rfp_margin_improving: [-5.70555],
    fut_margin_depth: [1.6328477, 26.032167],
    fut_margin_scalar: [53.843964],
    fut_margin_is_pv: [7.0337334],
    fut_margin_was_pv: [5.7520294],
    fut_margin_is_check: [7.6654572],
    fut_margin_is_killer: [6.9119344],
    fut_margin_improving: [4.4097533],
    fut_margin_gain: [1.2424359],
    nsp_margin_depth: [-2.1050088, -47.118065],
    nsp_margin_scalar: [8.325452],
    qsp_margin_depth: [-8.862688, -0.8800826],
    qsp_margin_scalar: [8.587346],
    sp_margin_is_killer: [-7.17011],
    lmp_depth: [0.4379748, 0.31323907],
    lmp_scalar: [0.9365009],
    lmp_baseline: [0.7023131],
    lmp_is_root: [1.5642166],
    lmp_is_pv: [1.125257],
    lmp_was_pv: [0.6963191],
    lmp_is_check: [0.9304215],
    lmp_improving: [0.9208485],
    lmr_depth: [0.0, 0.23456715, 0.10055269],
    lmr_index: [0.0, 0.09870157],
    lmr_scalar: [0.7319988],
    lmr_baseline: [0.3194101],
    lmr_is_root: [-0.25249255],
    lmr_is_pv: [-0.75320363],
    lmr_was_pv: [-0.2785572],
    lmr_gives_check: [-0.9013785],
    lmr_is_noisy_pv: [0.7057642],
    lmr_is_killer: [-1.23418],
    lmr_cut: [1.2753559],
    lmr_improving: [-0.40549764],
    lmr_history: [-0.85406846],
    lmr_counter: [-1.2011613],
    killer_rating: [53.16652],
    history_rating: [108.39066],
    counter_rating: [124.212105],
    winning_rating_margin: [-20.445045],
    winning_rating_gain: [0.0, 1.656793],
    winning_rating_scalar: [15.42193],
    aw_baseline: [2092.1055, 522.34143, 128.32942, 32.26601, 7.8898854, 4.9304986],
    aw_gamma: [1.4202363],
    aw_delta: [1.2325915],
    aw_fail_low_blend: [0.52635705],
    aw_fail_high_reduction: [1.0448276],
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

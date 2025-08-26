use std::{cell::SyncUnsafeCell, fmt::Debug};

#[cfg(feature = "spsa")]
use derive_more::with_trait::Display;

#[cfg(feature = "spsa")]
use ron::de::{SpannedError, from_str as deserialize};

#[cfg(feature = "spsa")]
use ron::ser::to_writer as serialize;

#[cfg(feature = "spsa")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "spsa")]
use std::fmt::{self, Formatter};

#[cfg(feature = "spsa")]
use std::str::FromStr;

#[cfg(feature = "spsa")]
mod variable;

#[cfg(feature = "spsa")]
pub use variable::*;

#[cfg(not(feature = "spsa"))]
mod constant;

#[cfg(not(feature = "spsa"))]
pub use constant::*;

pub trait Param: Sized {
    #[cfg(feature = "spsa")]
    const LEN: usize;

    type Value;

    fn get(&self) -> Self::Value;

    #[cfg(feature = "spsa")]
    fn min() -> Self::Value;

    #[cfg(feature = "spsa")]
    fn max() -> Self::Value;

    #[cfg(feature = "spsa")]
    fn perturb<I: IntoIterator<Item = f64>>(&self, perturbations: I) -> (Self, Self);

    #[cfg(feature = "spsa")]
    fn update<I: IntoIterator<Item = f64>>(&mut self, corrections: I);
}

static PARAMS: SyncUnsafeCell<Params> = SyncUnsafeCell::new(Params::new());

#[cfg(feature = "spsa")]
#[cold]
#[ctor::ctor]
#[inline(never)]
unsafe fn init() {
    Params::init(Default::default());
}

#[cfg(feature = "spsa")]
impl Display for Params {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
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
    ($name:ty,) => { 1 };
    ($first:ty, $($rest:ty,)*) => {
        <$first>::LEN + len!($($rest,)*)
    }
}

macro_rules! params {
    ($($name: ident: $type: ty,)*) => {
        #[derive(Debug, Default, Clone, PartialEq)]
        #[cfg_attr(test, derive(test_strategy::Arbitrary))]
        #[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
        #[cfg_attr(feature = "spsa", serde(deny_unknown_fields))]
        pub struct Params {
            $(#[cfg_attr(feature = "spsa", serde(default))] $name: $type,)*
        }

        impl Params {
            pub const BASE: i64 = 4096;

            const fn new() -> Self {
                Params {
                    $($name: <$type>::new(),)*
                }
            }
        }

        $(impl Params {
            /// This parameter's current value.
            #[inline(always)]
            pub fn $name() -> <$type as Param>::Value {
                unsafe { PARAMS.get().as_ref_unchecked().$name.get() }
            }
        })*

        #[cfg(feature = "spsa")]
        impl Params {
            /// The number of parameters.
            pub const LEN: usize = len!($($type,)*);

            /// Initializes the global [`PARAMS`].
            pub fn init(self) {
                unsafe { *PARAMS.get().as_mut_unchecked() = self }
            }

            /// Perturb parameters in both positive and negative directions.
            ///
            /// # Panic
            ///
            /// Panics if the `perturbations`'s length is less than [`Self::LEN`].
            pub fn perturb<I: IntoIterator<Item = f64>>(&self, perturbations: I) -> (Self, Self) {
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
            pub fn update<I: IntoIterator<Item = f64>>(&mut self, corrections: I) {
                let mut corrections = corrections.into_iter();
                $(self.$name.update(&mut corrections);)*
            }
        }
    };
}

params! {
    moves_left_start: Constant1<{ [726933] }>,
    moves_left_end: Constant1<{ [8548] }>,
    soft_time_fraction: Constant1<{ [3067] }>,
    hard_time_fraction: Constant1<{ [3501] }>,
    score_trend_inertia: Constant1<{ [28763] }>,
    pv_focus_gamma: Constant1<{ [6946] }>,
    pv_focus_delta: Constant1<{ [7757] }>,
    score_trend_magnitude: Constant1<{ [3547] }>,
    score_trend_pivot: Variable1<{ [116317] }>,
    pawn_values: Variable8<{ [170087, 212243, 216138, 235716, 237581, 233960, 222476, 206288] }>,
    knight_values: Variable8<{ [949589, 754144, 779567, 767893, 790824, 759921, 755225, 693866] }>,
    bishop_values: Variable8<{ [997239, 859957, 865114, 904453, 894078, 853504, 859904, 786722] }>,
    rook_values: Variable8<{ [1017734, 1282707, 1332641, 1359395, 1342194, 1293143, 1300675, 1200942] }>,
    queen_values: Variable8<{ [2277089, 2409262, 2615464, 2674716, 2498073, 2577281, 2563294, 2634654] }>,
    pawns_correction: Variable8<{ [57611, 56665, 56889, 56357, 57382, 56082, 57957, 56843] }>,
    minor_correction: Variable8<{ [40887, 40700, 39683, 40568, 40514, 39941, 39991, 39609] }>,
    major_correction: Variable8<{ [45978, 45124, 45678, 46153, 46242, 46512, 46214, 46944] }>,
    pieces_correction: Variable8<{ [66928, 66677, 68208, 66884, 67095, 68777, 66682, 66350] }>,
    correction_gradient_depth: Variable2<{ [0, 2016] }>,
    correction_gradient_scalar: Variable1<{ [2621] }>,
    pawns_correction_bonus: Variable1<{ [7555] }>,
    minor_correction_bonus: Variable1<{ [7477] }>,
    major_correction_bonus: Variable1<{ [7756] }>,
    pieces_correction_bonus: Variable1<{ [7692] }>,
    history_bonus_depth: Variable2<{ [0, 20843] }>,
    history_bonus_scalar: Variable1<{ [3018] }>,
    continuation_bonus_depth: Variable2<{ [0, 27082] }>,
    continuation_bonus_scalar: Variable1<{ [3575] }>,
    history_penalty_depth: Variable2<{ [0, -24159] }>,
    history_penalty_scalar: Variable1<{ [-6348] }>,
    continuation_penalty_depth: Variable2<{ [0, -22603] }>,
    continuation_penalty_scalar: Variable1<{ [-2919] }>,
    probcut_margin_depth: Variable2<{ [0, 55497] }>,
    probcut_margin_scalar: Variable1<{ [838815] }>,
    single_extension_margin_depth: Variable2<{ [0, 3002] }>,
    single_extension_margin_scalar: Variable1<{ [2213] }>,
    double_extension_margin_depth: Variable2<{ [0, 4493] }>,
    double_extension_margin_scalar: Variable1<{ [1079] }>,
    triple_extension_margin_depth: Variable2<{ [0, 2400] }>,
    triple_extension_margin_scalar: Variable1<{ [586953] }>,
    null_move_reduction_gamma: Variable1<{ [37402] }>,
    null_move_reduction_delta: Variable1<{ [21993] }>,
    null_move_pruning_depth: Variable2<{ [2174, 40772] }>,
    null_move_pruning_scalar: Variable1<{ [-20082] }>,
    fail_low_pruning_depth: Variable2<{ [97704, 1390978] }>,
    fail_low_pruning_scalar: Variable1<{ [-530477] }>,
    fail_high_pruning_depth: Variable2<{ [133635, 261459] }>,
    fail_high_pruning_scalar: Variable1<{ [-211458] }>,
    razoring_margin_depth: Variable2<{ [33819, 89868] }>,
    razoring_margin_scalar: Variable1<{ [134292] }>,
    reverse_futility_margin_depth: Variable2<{ [5078, 52206] }>,
    reverse_futility_margin_scalar: Variable1<{ [37380] }>,
    reverse_futility_margin_improving: Variable1<{ [-23217] }>,
    futility_margin_depth: Variable2<{ [7117, 107105] }>,
    futility_margin_scalar: Variable1<{ [214102] }>,
    futility_margin_is_pv: Variable1<{ [29292] }>,
    futility_margin_was_pv: Variable1<{ [24250] }>,
    futility_margin_is_check: Variable1<{ [31351] }>,
    futility_margin_is_killer: Variable1<{ [27640] }>,
    futility_margin_improving: Variable1<{ [17497] }>,
    futility_margin_gain: Variable1<{ [4769] }>,
    noisy_see_pruning_depth: Variable2<{ [-8892, -189868] }>,
    noisy_see_pruning_scalar: Variable1<{ [35940] }>,
    quiet_see_pruning_depth: Variable2<{ [-36999, -3516] }>,
    quiet_see_pruning_scalar: Variable1<{ [34218] }>,
    see_pruning_is_killer: Variable1<{ [-28451] }>,
    late_move_reduction_depth: Variable3<{ [0, 935, 396] }>,
    late_move_reduction_index: Variable2<{ [0, 403] }>,
    late_move_reduction_scalar: Variable1<{ [3125] }>,
    late_move_reduction_baseline: Variable1<{ [1311] }>,
    late_move_reduction_is_root: Variable1<{ [-1060] }>,
    late_move_reduction_is_pv: Variable1<{ [-3075] }>,
    late_move_reduction_was_pv: Variable1<{ [-1085] }>,
    late_move_reduction_gives_check: Variable1<{ [-3737] }>,
    late_move_reduction_is_noisy_pv: Variable1<{ [2952] }>,
    late_move_reduction_is_killer: Variable1<{ [-5078] }>,
    late_move_reduction_cut: Variable1<{ [5229] }>,
    late_move_reduction_improving: Variable1<{ [-1511] }>,
    late_move_reduction_history: Variable1<{ [-3609] }>,
    late_move_reduction_counter: Variable1<{ [-4795] }>,
    late_move_pruning_depth: Variable2<{ [1783, 1301] }>,
    late_move_pruning_scalar: Variable1<{ [3716] }>,
    late_move_pruning_baseline: Variable1<{ [2793] }>,
    late_move_pruning_is_root: Variable1<{ [6536] }>,
    late_move_pruning_is_pv: Variable1<{ [4696] }>,
    late_move_pruning_was_pv: Variable1<{ [2874] }>,
    late_move_pruning_is_check: Variable1<{ [3810] }>,
    late_move_pruning_improving: Variable1<{ [3903] }>,
    improving_2: Variable2<{ [3630, 0] }>,
    improving_4: Variable1<{ [3662] }>,
    killer_move_bonus: Variable1<{ [213899] }>,
    history_rating: Variable1<{ [427913] }>,
    counter_rating: Variable1<{ [528866] }>,
    winning_rating_margin: Variable1<{ [-83206] }>,
    winning_rating_gain: Variable2<{ [0, 6795] }>,
    winning_rating_scalar: Variable1<{ [66741] }>,
    aspiration_window_baseline: Variable1<{ [23022] }>,
    aspiration_window_exponent: Variable2<{ [0, 5857] }>,
    aspiration_window_scalar: Variable1<{ [6403] }>,
}

#[cfg(test)]
#[cfg(feature = "spsa")]
mod tests {
    use super::*;
    use proptest::sample::size_range;
    use test_strategy::proptest;

    #[proptest]
    fn perturbing_updates_params(p: Params, #[any(size_range(Params::LEN).lift())] d: Vec<f64>) {
        let (mut l, mut r) = (p.clone(), p.clone());
        l.update(d.iter().copied());
        r.update(d.iter().map(|&d| -d));
        assert_eq!(p.perturb(d), (l, r));
    }

    #[proptest]
    fn parsing_printed_params_is_an_identity(p: Params) {
        assert_eq!(p.to_string().parse(), Ok(p));
    }
}

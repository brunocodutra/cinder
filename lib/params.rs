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
    moves_left_start: Constant1<{ [714103] }>,
    moves_left_end: Constant1<{ [8548] }>,
    soft_time_fraction: Constant1<{ [3079] }>,
    hard_time_fraction: Constant1<{ [3507] }>,
    score_trend_inertia: Constant1<{ [28464] }>,
    pv_focus_gamma: Constant1<{ [6874] }>,
    pv_focus_delta: Constant1<{ [7814] }>,
    score_trend_magnitude: Constant1<{ [3515] }>,
    score_trend_pivot: Variable1<{ [113529] }>,
    pawn_values: Variable8<{ [167605, 208220, 213271, 234805, 237347, 230929, 223893, 208501] }>,
    knight_values: Variable8<{ [947219, 762688, 780281, 767472, 802729, 768939, 747888, 692013] }>,
    bishop_values: Variable8<{ [981463, 876610, 857956, 899411, 899683, 847332, 849839, 789404] }>,
    rook_values: Variable8<{ [1012391, 1264833, 1334871, 1360648, 1368886, 1273232, 1291030, 1208929] }>,
    queen_values: Variable8<{ [2244126, 2438351, 2612095, 2665591, 2562592, 2569049, 2545001, 2620808] }>,
    pawns_correction: Variable1<{ [57117] }>,
    minor_correction: Variable1<{ [40252] }>,
    major_correction: Variable1<{ [46560] }>,
    pieces_correction: Variable1<{ [67092] }>,
    correction_gradient_depth: Variable2<{ [0, 1981] }>,
    correction_gradient_scalar: Variable1<{ [2572] }>,
    pawns_correction_bonus: Variable1<{ [7661] }>,
    minor_correction_bonus: Variable1<{ [7358] }>,
    major_correction_bonus: Variable1<{ [7684] }>,
    pieces_correction_bonus: Variable1<{ [7723] }>,
    history_bonus_depth: Variable2<{ [0, 20919] }>,
    history_bonus_scalar: Variable1<{ [3010] }>,
    continuation_bonus_depth: Variable2<{ [0, 27004] }>,
    continuation_bonus_scalar: Variable1<{ [3558] }>,
    history_penalty_depth: Variable2<{ [0, -23435] }>,
    history_penalty_scalar: Variable1<{ [-6362] }>,
    continuation_penalty_depth: Variable2<{ [0, -23132] }>,
    continuation_penalty_scalar: Variable1<{ [-2976] }>,
    probcut_margin_depth: Variable2<{ [0, 55573] }>,
    probcut_margin_scalar: Variable1<{ [843773] }>,
    single_extension_margin_depth: Variable2<{ [0, 2939] }>,
    single_extension_margin_scalar: Variable1<{ [2198] }>,
    double_extension_margin_depth: Variable2<{ [0, 4488] }>,
    double_extension_margin_scalar: Variable1<{ [1085] }>,
    triple_extension_margin_depth: Variable2<{ [0, 2423] }>,
    triple_extension_margin_scalar: Variable1<{ [588940] }>,
    null_move_reduction_gamma: Variable1<{ [37480] }>,
    null_move_reduction_delta: Variable1<{ [22449] }>,
    null_move_pruning_depth: Variable2<{ [2179, 40184] }>,
    null_move_pruning_scalar: Variable1<{ [-20046] }>,
    fail_low_pruning_depth: Variable2<{ [95408, 1359932] }>,
    fail_low_pruning_scalar: Variable1<{ [-528383] }>,
    fail_high_pruning_depth: Variable2<{ [132577, 263190] }>,
    fail_high_pruning_scalar: Variable1<{ [-214063] }>,
    razoring_margin_depth: Variable2<{ [33747, 91028] }>,
    razoring_margin_scalar: Variable1<{ [135957] }>,
    reverse_futility_margin_depth: Variable2<{ [5121, 52212] }>,
    reverse_futility_margin_scalar: Variable1<{ [36870] }>,
    reverse_futility_margin_improving: Variable1<{ [-23379] }>,
    futility_margin_depth: Variable2<{ [6981, 105833] }>,
    futility_margin_scalar: Variable1<{ [210090] }>,
    futility_margin_is_pv: Variable1<{ [29177] }>,
    futility_margin_was_pv: Variable1<{ [24485] }>,
    futility_margin_is_check: Variable1<{ [31510] }>,
    futility_margin_is_killer: Variable1<{ [27589] }>,
    futility_margin_improving: Variable1<{ [17695] }>,
    futility_margin_gain: Variable1<{ [4744] }>,
    noisy_see_pruning_depth: Variable2<{ [-8927, -188260] }>,
    noisy_see_pruning_scalar: Variable1<{ [36145] }>,
    quiet_see_pruning_depth: Variable2<{ [-37091, -3543] }>,
    quiet_see_pruning_scalar: Variable1<{ [34777] }>,
    see_pruning_is_killer: Variable1<{ [-28302] }>,
    late_move_reduction_depth: Variable3<{ [0, 951, 393] }>,
    late_move_reduction_index: Variable2<{ [0, 398] }>,
    late_move_reduction_scalar: Variable1<{ [3100] }>,
    late_move_reduction_baseline: Variable1<{ [1327] }>,
    late_move_reduction_is_root: Variable1<{ [-1048] }>,
    late_move_reduction_is_pv: Variable1<{ [-3084] }>,
    late_move_reduction_was_pv: Variable1<{ [-1087] }>,
    late_move_reduction_gives_check: Variable1<{ [-3691] }>,
    late_move_reduction_is_noisy_pv: Variable1<{ [2891] }>,
    late_move_reduction_is_killer: Variable1<{ [-4990] }>,
    late_move_reduction_cut: Variable1<{ [5310] }>,
    late_move_reduction_improving: Variable1<{ [-1520] }>,
    late_move_reduction_history: Variable1<{ [-3637] }>,
    late_move_reduction_counter: Variable1<{ [-4886] }>,
    late_move_pruning_depth: Variable2<{ [1766, 1300] }>,
    late_move_pruning_scalar: Variable1<{ [3760] }>,
    late_move_pruning_baseline: Variable1<{ [2862] }>,
    late_move_pruning_is_root: Variable1<{ [6485] }>,
    late_move_pruning_is_pv: Variable1<{ [4695] }>,
    late_move_pruning_was_pv: Variable1<{ [2955] }>,
    late_move_pruning_is_check: Variable1<{ [3770] }>,
    late_move_pruning_improving: Variable1<{ [3915] }>,
    improving_2: Variable2<{ [3612, 0] }>,
    improving_4: Variable1<{ [3737] }>,
    killer_move_bonus: Variable1<{ [212760] }>,
    history_rating: Variable1<{ [426635] }>,
    counter_rating: Variable1<{ [532230] }>,
    winning_rating_margin: Variable1<{ [-81614] }>,
    winning_rating_gain: Variable2<{ [0, 6868] }>,
    winning_rating_scalar: Variable1<{ [66637] }>,
    aspiration_window_baseline: Variable1<{ [22785] }>,
    aspiration_window_exponent: Variable2<{ [0, 5850] }>,
    aspiration_window_scalar: Variable1<{ [6346] }>,
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

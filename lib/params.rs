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
    value_scale: Constant1<{ [524288] }>,
    moves_left_start: Constant1<{ [729101] }>,
    moves_left_end: Constant1<{ [8541] }>,
    soft_time_fraction: Constant1<{ [3055] }>,
    hard_time_fraction: Constant1<{ [3501] }>,
    score_trend_inertia: Constant1<{ [28907] }>,
    pv_focus_gamma: Constant1<{ [6752] }>,
    pv_focus_delta: Constant1<{ [7735] }>,
    score_trend_magnitude: Constant1<{ [3439] }>,
    score_trend_pivot: Variable1<{ [136652] }>,
    pawns_correction: Variable1<{ [53979] }>,
    minor_correction: Variable1<{ [43146] }>,
    major_correction: Variable1<{ [54277] }>,
    pieces_correction: Variable1<{ [65508] }>,
    correction_gradient_depth: Variable2<{ [0, 2158] }>,
    correction_gradient_scalar: Variable1<{ [2655] }>,
    pawns_correction_bonus: Variable1<{ [7178] }>,
    minor_correction_bonus: Variable1<{ [7021] }>,
    major_correction_bonus: Variable1<{ [7319] }>,
    pieces_correction_bonus: Variable1<{ [6929] }>,
    history_bonus_depth: Variable3<{ [0, 2959, 6946] }>,
    history_bonus_is_quiet: Variable2<{ [0, 1046] }>,
    history_bonus_scalar: Variable1<{ [386] }>,
    continuation_bonus_depth: Variable3<{ [0, 1120, 9832] }>,
    continuation_bonus_is_quiet: Variable2<{ [0, 226] }>,
    continuation_bonus_scalar: Variable1<{ [1552] }>,
    history_penalty_depth: Variable3<{ [0, -2039, -8320] }>,
    history_penalty_is_quiet: Variable2<{ [0, -1382] }>,
    history_penalty_scalar: Variable1<{ [-2269] }>,
    continuation_penalty_depth: Variable3<{ [0, -636, -9410] }>,
    continuation_penalty_is_quiet: Variable2<{ [0, 546] }>,
    continuation_penalty_scalar: Variable1<{ [-1772] }>,
    null_move_reduction_gamma: Variable1<{ [47592] }>,
    null_move_reduction_delta: Variable1<{ [29931] }>,
    null_move_pruning_depth: Variable2<{ [2000, 40000] }>,
    null_move_pruning_scalar: Variable1<{ [-20000] }>,
    fail_low_pruning_depth: Variable2<{ [95365, 1321811] }>,
    fail_low_pruning_scalar: Variable1<{ [-492249] }>,
    fail_high_pruning_depth: Variable2<{ [126088, 233550] }>,
    fail_high_pruning_scalar: Variable1<{ [-206953] }>,
    razoring_margin_depth: Variable2<{ [33040, 103495] }>,
    razoring_margin_scalar: Variable1<{ [156499] }>,
    reverse_futility_margin_depth: Variable2<{ [5696, 43718] }>,
    reverse_futility_margin_scalar: Variable1<{ [36429] }>,
    reverse_futility_margin_improving: Variable1<{ [-21818] }>,
    reverse_futility_margin_is_noisy_pv: Variable1<{ [-4783] }>,
    reverse_futility_margin_cut: Variable1<{ [-11229] }>,
    futility_margin_depth: Variable2<{ [7306, 105353] }>,
    futility_margin_scalar: Variable1<{ [243827] }>,
    futility_margin_is_pv: Variable1<{ [31483] }>,
    futility_margin_was_pv: Variable1<{ [25958] }>,
    futility_margin_is_check: Variable1<{ [29237] }>,
    futility_margin_is_killer: Variable1<{ [27284] }>,
    futility_margin_improving: Variable1<{ [15524] }>,
    futility_margin_gain: Variable1<{ [5587] }>,
    single_extension_margin_depth: Variable2<{ [0, 3074] }>,
    single_extension_margin_scalar: Variable1<{ [2013] }>,
    double_extension_margin_depth: Variable2<{ [0, 4813] }>,
    double_extension_margin_scalar: Variable1<{ [1214] }>,
    triple_extension_margin_depth: Variable2<{ [0, 2501] }>,
    triple_extension_margin_scalar: Variable1<{ [608566] }>,
    see_pruning_depth: Variable2<{ [-19987, -203543] }>,
    see_pruning_scalar: Variable1<{ [4054] }>,
    see_pruning_is_killer: Variable1<{ [-31405] }>,
    see_pruning_history: Variable1<{ [-10913] }>,
    see_pruning_counter: Variable1<{ [-21753] }>,
    late_move_reduction_depth: Variable3<{ [0, 917, 381] }>,
    late_move_reduction_index: Variable2<{ [0, 418] }>,
    late_move_reduction_scalar: Variable1<{ [3150] }>,
    late_move_reduction_baseline: Variable1<{ [1127] }>,
    late_move_reduction_is_root: Variable1<{ [-1166] }>,
    late_move_reduction_is_pv: Variable1<{ [-2989] }>,
    late_move_reduction_was_pv: Variable1<{ [-933] }>,
    late_move_reduction_gives_check: Variable1<{ [-3703] }>,
    late_move_reduction_is_noisy_pv: Variable1<{ [3255] }>,
    late_move_reduction_is_killer: Variable1<{ [-4602] }>,
    late_move_reduction_cut: Variable1<{ [5513] }>,
    late_move_reduction_improving: Variable1<{ [-1692] }>,
    late_move_reduction_history: Variable1<{ [-3837] }>,
    late_move_reduction_counter: Variable1<{ [-5762] }>,
    late_move_pruning_depth: Variable2<{ [1796, 1193] }>,
    late_move_pruning_scalar: Variable1<{ [4017] }>,
    late_move_pruning_baseline: Variable1<{ [3282] }>,
    late_move_pruning_is_root: Variable1<{ [7099] }>,
    late_move_pruning_is_pv: Variable1<{ [4302] }>,
    late_move_pruning_was_pv: Variable1<{ [2974] }>,
    late_move_pruning_is_check: Variable1<{ [3673] }>,
    late_move_pruning_improving: Variable1<{ [4506] }>,
    improving_2: Variable2<{ [3769, 0] }>,
    improving_4: Variable1<{ [4689] }>,
    killer_move_bonus: Variable1<{ [215468] }>,
    history_rating: Variable1<{ [428682] }>,
    counter_rating: Variable1<{ [665772] }>,
    winning_rating_depth: Variable2<{ [0, 6634] }>,
    winning_rating_scalar: Variable1<{ [60555] }>,
    aspiration_window_baseline: Variable1<{ [22891] }>,
    aspiration_window_exponent: Variable2<{ [0, 5857] }>,
    aspiration_window_scalar: Variable1<{ [6485] }>,
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

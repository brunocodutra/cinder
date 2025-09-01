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
    score_trend_pivot: Variable1<{ [115952] }>,
    pawns_correction: Variable1<{ [59011] }>,
    minor_correction: Variable1<{ [40285] }>,
    major_correction: Variable1<{ [52408] }>,
    pieces_correction: Variable1<{ [66331] }>,
    correction_gradient_depth: Variable2<{ [0, 2040] }>,
    correction_gradient_scalar: Variable1<{ [2661] }>,
    pawns_correction_bonus: Variable1<{ [7303] }>,
    minor_correction_bonus: Variable1<{ [6901] }>,
    major_correction_bonus: Variable1<{ [7369] }>,
    pieces_correction_bonus: Variable1<{ [6881] }>,
    history_bonus_depth: Variable2<{ [0, 20658] }>,
    history_bonus_scalar: Variable1<{ [2932] }>,
    continuation_bonus_depth: Variable2<{ [0, 25204] }>,
    continuation_bonus_scalar: Variable1<{ [3528] }>,
    history_penalty_depth: Variable2<{ [0, -25204] }>,
    history_penalty_scalar: Variable1<{ [-7056] }>,
    continuation_penalty_depth: Variable2<{ [0, -21640] }>,
    continuation_penalty_scalar: Variable1<{ [-2968] }>,
    single_extension_margin_depth: Variable2<{ [0, 3267] }>,
    single_extension_margin_scalar: Variable1<{ [2141] }>,
    double_extension_margin_depth: Variable2<{ [0, 4246] }>,
    double_extension_margin_scalar: Variable1<{ [1111] }>,
    triple_extension_margin_depth: Variable2<{ [0, 2295] }>,
    triple_extension_margin_scalar: Variable1<{ [575866] }>,
    null_move_reduction_gamma: Variable1<{ [42716] }>,
    null_move_reduction_delta: Variable1<{ [23984] }>,
    null_move_pruning_depth: Variable2<{ [2070, 39073] }>,
    null_move_pruning_scalar: Variable1<{ [-21414] }>,
    fail_low_pruning_depth: Variable2<{ [92281, 1435995] }>,
    fail_low_pruning_scalar: Variable1<{ [-519260] }>,
    fail_high_pruning_depth: Variable2<{ [127849, 250772] }>,
    fail_high_pruning_scalar: Variable1<{ [-193364] }>,
    razoring_margin_depth: Variable2<{ [32631, 97981] }>,
    razoring_margin_scalar: Variable1<{ [151486] }>,
    reverse_futility_margin_depth: Variable2<{ [5250, 48192] }>,
    reverse_futility_margin_scalar: Variable1<{ [36518] }>,
    reverse_futility_margin_improving: Variable1<{ [-22725] }>,
    futility_margin_depth: Variable2<{ [7783, 116394] }>,
    futility_margin_scalar: Variable1<{ [241533] }>,
    futility_margin_is_pv: Variable1<{ [29737] }>,
    futility_margin_was_pv: Variable1<{ [27294] }>,
    futility_margin_is_check: Variable1<{ [29874] }>,
    futility_margin_is_killer: Variable1<{ [27493] }>,
    futility_margin_improving: Variable1<{ [16884] }>,
    futility_margin_gain: Variable1<{ [5035] }>,
    noisy_see_pruning_depth: Variable2<{ [-9244, -202042] }>,
    noisy_see_pruning_scalar: Variable1<{ [35605] }>,
    quiet_see_pruning_depth: Variable2<{ [-40119, -4118] }>,
    quiet_see_pruning_scalar: Variable1<{ [36777] }>,
    see_pruning_is_killer: Variable1<{ [-26725] }>,
    late_move_reduction_depth: Variable3<{ [0, 947, 378] }>,
    late_move_reduction_index: Variable2<{ [0, 401] }>,
    late_move_reduction_scalar: Variable1<{ [3145] }>,
    late_move_reduction_baseline: Variable1<{ [1305] }>,
    late_move_reduction_is_root: Variable1<{ [-1042] }>,
    late_move_reduction_is_pv: Variable1<{ [-3031] }>,
    late_move_reduction_was_pv: Variable1<{ [-1010] }>,
    late_move_reduction_gives_check: Variable1<{ [-3710] }>,
    late_move_reduction_is_noisy_pv: Variable1<{ [3072] }>,
    late_move_reduction_is_killer: Variable1<{ [-5053] }>,
    late_move_reduction_cut: Variable1<{ [5431] }>,
    late_move_reduction_improving: Variable1<{ [-1611] }>,
    late_move_reduction_history: Variable1<{ [-3374] }>,
    late_move_reduction_counter: Variable1<{ [-5773] }>,
    late_move_pruning_depth: Variable2<{ [1899, 1300] }>,
    late_move_pruning_scalar: Variable1<{ [3780] }>,
    late_move_pruning_baseline: Variable1<{ [3310] }>,
    late_move_pruning_is_root: Variable1<{ [6575] }>,
    late_move_pruning_is_pv: Variable1<{ [4659] }>,
    late_move_pruning_was_pv: Variable1<{ [2984] }>,
    late_move_pruning_is_check: Variable1<{ [3708] }>,
    late_move_pruning_improving: Variable1<{ [4153] }>,
    improving_2: Variable2<{ [3601, 0] }>,
    improving_4: Variable1<{ [4462] }>,
    killer_move_bonus: Variable1<{ [214389] }>,
    history_rating: Variable1<{ [434541] }>,
    counter_rating: Variable1<{ [618791] }>,
    winning_rating_depth: Variable2<{ [0, 6992] }>,
    winning_rating_scalar: Variable1<{ [63469] }>,
    aspiration_window_baseline: Variable1<{ [22609] }>,
    aspiration_window_exponent: Variable2<{ [0, 6199] }>,
    aspiration_window_scalar: Variable1<{ [5988] }>,
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

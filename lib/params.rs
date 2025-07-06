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
            pub const BASE: i32 = 4096;

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
    score_trend_pivot: Variable1<{ [128037] }>,
    pawns_correction: Variable1<{ [53122] }>,
    minor_correction: Variable1<{ [42507] }>,
    major_correction: Variable1<{ [53283] }>,
    pieces_correction: Variable1<{ [63172] }>,
    correction_gradient_gamma: Variable1<{ [2197] }>,
    correction_gradient_delta: Variable1<{ [2816] }>,
    pawns_correction_bonus: Variable1<{ [110336] }>,
    minor_correction_bonus: Variable1<{ [116576] }>,
    major_correction_bonus: Variable1<{ [119200] }>,
    pieces_correction_bonus: Variable1<{ [111584] }>,
    quiet_history_bonus_gamma: Variable1<{ [9242] }>,
    quiet_history_bonus_delta: Variable1<{ [1394] }>,
    noisy_history_bonus_gamma: Variable1<{ [6489] }>,
    noisy_history_bonus_delta: Variable1<{ [385] }>,
    quiet_continuation_bonus_gamma: Variable1<{ [10262] }>,
    quiet_continuation_bonus_delta: Variable1<{ [1772] }>,
    noisy_continuation_bonus_gamma: Variable1<{ [9174] }>,
    noisy_continuation_bonus_delta: Variable1<{ [1553] }>,
    quiet_history_penalty_gamma: Variable1<{ [-10964] }>,
    quiet_history_penalty_delta: Variable1<{ [-3622] }>,
    noisy_history_penalty_gamma: Variable1<{ [-8954] }>,
    noisy_history_penalty_delta: Variable1<{ [-2331] }>,
    quiet_continuation_penalty_gamma: Variable1<{ [-10082] }>,
    quiet_continuation_penalty_delta: Variable1<{ [-1355] }>,
    noisy_continuation_penalty_gamma: Variable1<{ [-9464] }>,
    noisy_continuation_penalty_delta: Variable1<{ [-1900] }>,
    fail_high_reduction_gamma: Variable1<{ [469832] }>,
    fail_high_reduction_delta: Variable1<{ [234248] }>,
    fail_low_reduction_gamma: Variable1<{ [1473904] }>,
    fail_low_reduction_delta: Variable1<{ [481719] }>,
    single_extension_margin_gamma: Variable1<{ [2893] }>,
    single_extension_margin_delta: Variable1<{ [2059] }>,
    double_extension_margin_gamma: Variable1<{ [4826] }>,
    double_extension_margin_delta: Variable1<{ [1173] }>,
    triple_extension_margin_gamma: Variable1<{ [2600] }>,
    triple_extension_margin_delta: Variable1<{ [636415] }>,
    null_move_pruning_gamma: Variable1<{ [47917] }>,
    null_move_pruning_delta: Variable1<{ [29724] }>,
    razoring_margin_theta: Variable1<{ [39340] }>,
    razoring_margin_gamma: Variable1<{ [97390] }>,
    razoring_margin_delta: Variable1<{ [154303] }>,
    reverse_futility_margin_theta: Variable1<{ [7786] }>,
    reverse_futility_margin_gamma: Variable1<{ [41220] }>,
    reverse_futility_margin_delta: Variable1<{ [36629] }>,
    reverse_futility_margin_improving: Variable1<{ [-21363] }>,
    reverse_futility_margin_noisy_pv: Variable1<{ [-4902] }>,
    reverse_futility_margin_cut: Variable1<{ [-11169] }>,
    futility_margin_theta: Variable1<{ [1541] }>,
    futility_margin_gamma: Variable1<{ [133928] }>,
    futility_margin_delta: Variable1<{ [232438] }>,
    futility_margin_is_pv: Variable1<{ [30176] }>,
    futility_margin_was_pv: Variable1<{ [26686] }>,
    futility_margin_improving: Variable1<{ [16346] }>,
    futility_margin_gain: Variable1<{ [5710] }>,
    futility_margin_killer: Variable1<{ [27863] }>,
    futility_margin_check: Variable1<{ [27551] }>,
    futility_margin_history: Variable1<{ [18111] }>,
    futility_margin_counter: Variable1<{ [20278] }>,
    see_pruning_theta: Variable1<{ [-2392] }>,
    see_pruning_gamma: Variable1<{ [-249649] }>,
    see_pruning_delta: Variable1<{ [3922] }>,
    see_pruning_killer: Variable1<{ [-31273] }>,
    see_pruning_history: Variable1<{ [-10874] }>,
    see_pruning_counter: Variable1<{ [-21552] }>,
    late_move_reduction_theta: Variable1<{ [918] }>,
    late_move_reduction_gamma: Variable1<{ [398] }>,
    late_move_reduction_delta: Variable1<{ [2919] }>,
    late_move_reduction_baseline: Variable1<{ [1219] }>,
    late_move_reduction_root: Variable1<{ [-2384] }>,
    late_move_reduction_is_pv: Variable1<{ [-2906] }>,
    late_move_reduction_was_pv: Variable1<{ [-901] }>,
    late_move_reduction_improving: Variable1<{ [-1763] }>,
    late_move_reduction_killer: Variable1<{ [-4802] }>,
    late_move_reduction_check: Variable1<{ [-3988] }>,
    late_move_reduction_history: Variable1<{ [-4127] }>,
    late_move_reduction_counter: Variable1<{ [-5773] }>,
    late_move_reduction_noisy_pv: Variable1<{ [3242] }>,
    late_move_reduction_cut: Variable1<{ [5715] }>,
    late_move_pruning_theta: Variable1<{ [1772] }>,
    late_move_pruning_gamma: Variable1<{ [1160] }>,
    late_move_pruning_delta: Variable1<{ [4131] }>,
    late_move_pruning_baseline: Variable1<{ [3056] }>,
    late_move_pruning_root: Variable1<{ [4422] }>,
    late_move_pruning_is_pv: Variable1<{ [4235] }>,
    late_move_pruning_was_pv: Variable1<{ [3281] }>,
    late_move_pruning_improving: Variable1<{ [4657] }>,
    late_move_pruning_check: Variable1<{ [3917] }>,
    killer_move_bonus: Variable1<{ [225177] }>,
    history_rating: Variable1<{ [437376] }>,
    counter_rating: Variable1<{ [680320] }>,
    winning_rating_gamma: Variable1<{ [7102] }>,
    winning_rating_delta: Variable1<{ [61864] }>,
    aspiration_window_start: Variable1<{ [22194] }>,
    aspiration_window_gamma: Variable1<{ [5599] }>,
    aspiration_window_delta: Variable1<{ [6676] }>,
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

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
    ($name:ident,) => { 1 };
    ($first:ident, $($rest:ident,)*) => {
        1 + len!($($rest,)*)
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
            pub fn $name() -> i32 {
                unsafe { PARAMS.get().as_ref_unchecked().$name.get() }
            }
        })*

        #[cfg(feature = "spsa")]
        impl Params {
            /// The number of parameters.
            pub const LEN: usize = len!($($name,)*);

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
    value_scale: Param<524288, 0>,
    moves_left_start: Param<736629>,
    moves_left_end: Param<9098>,
    soft_time_fraction: Param<2829>,
    hard_time_fraction: Param<3258>,
    score_trend_inertia: Param<26635>,
    pv_focus_gamma: Param<6909>,
    pv_focus_delta: Param<7832>,
    score_trend_magnitude: Param<3093>,
    score_trend_pivot: Param<205109>,
    quiet_history_bonus_gamma: Param<6965>,
    quiet_history_bonus_delta: Param<1271>,
    noisy_history_bonus_gamma: Param<6863>,
    noisy_history_bonus_delta: Param<1076>,
    quiet_continuation_bonus_gamma: Param<7993>,
    quiet_continuation_bonus_delta: Param<298>,
    noisy_continuation_bonus_gamma: Param<8574>,
    noisy_continuation_bonus_delta: Param<1476>,
    quiet_history_penalty_gamma: Param<9126>,
    quiet_history_penalty_delta: Param<2208>,
    noisy_history_penalty_gamma: Param<7365>,
    noisy_history_penalty_delta: Param<2266>,
    quiet_continuation_penalty_gamma: Param<8448>,
    quiet_continuation_penalty_delta: Param<1223>,
    noisy_continuation_penalty_gamma: Param<7969>,
    noisy_continuation_penalty_delta: Param<515>,
    null_move_reduction_gamma: Param<56651>,
    null_move_reduction_delta: Param<39193>,
    fail_high_reduction_gamma: Param<437596>,
    fail_high_reduction_delta: Param<236117>,
    fail_low_reduction_gamma: Param<1582341>,
    fail_low_reduction_delta: Param<362258>,
    single_extension_margin_gamma: Param<3040>,
    single_extension_margin_delta: Param<1391>,
    double_extension_margin_gamma: Param<4276>,
    double_extension_margin_delta: Param<1398>,
    triple_extension_margin_gamma: Param<2838>,
    triple_extension_margin_delta: Param<553652>,
    razoring_margin_gamma: Param<254471>,
    razoring_margin_delta: Param<164557>,
    reverse_futility_margin_gamma: Param<60237>,
    reverse_futility_margin_delta: Param<31144>,
    futility_margin_gamma: Param<162269>,
    futility_margin_delta: Param<266251>,
    futility_pruning_threshold: Param<40055>,
    see_pruning_threshold: Param<250703>,
    late_move_reduction_gamma: Param<1284>,
    late_move_reduction_delta: Param<3102>,
    late_move_reduction_pv: Param<4352>,
    late_move_reduction_cut: Param<4627>,
    late_move_reduction_improving: Param<1157>,
    late_move_reduction_history: Param<4701>,
    late_move_reduction_continuation: Param<4383>,
    late_move_reduction_killer: Param<4075>,
    late_move_reduction_check: Param<3638>,
    late_move_reduction_tt_noisy: Param<4002>,
    late_move_pruning_gamma: Param<2049>,
    late_move_pruning_delta: Param<4469>,
    killer_move_bonus: Param<234000>,
    history_rating: Param<4504>,
    continuation_rating: Param<5827>,
    winning_rating_gamma: Param<6643>,
    winning_rating_delta: Param<60372>,
    aspiration_window_start: Param<17850>,
    aspiration_window_gamma: Param<5164>,
    aspiration_window_delta: Param<6274>,
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

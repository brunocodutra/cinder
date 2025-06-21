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
    moves_left_start: Param<697292>,
    moves_left_end: Param<9788>,
    soft_time_fraction: Param<2464>,
    hard_time_fraction: Param<3274>,
    score_trend_inertia: Param<26033>,
    pv_focus_gamma: Param<7796>,
    pv_focus_delta: Param<8800>,
    score_trend_magnitude: Param<2316>,
    score_trend_pivot: Param<231467>,
    quiet_history_bonus_gamma: Param<6312>,
    quiet_history_bonus_delta: Param<1161>,
    noisy_history_bonus_gamma: Param<6772>,
    noisy_history_bonus_delta: Param<1168>,
    quiet_continuation_bonus_gamma: Param<7406>,
    quiet_continuation_bonus_delta: Param<508>,
    noisy_continuation_bonus_gamma: Param<7633>,
    noisy_continuation_bonus_delta: Param<1139>,
    quiet_history_penalty_gamma: Param<7879>,
    quiet_history_penalty_delta: Param<1781>,
    noisy_history_penalty_gamma: Param<6629>,
    noisy_history_penalty_delta: Param<2123>,
    quiet_continuation_penalty_gamma: Param<7343>,
    quiet_continuation_penalty_delta: Param<990>,
    noisy_continuation_penalty_gamma: Param<7812>,
    noisy_continuation_penalty_delta: Param<553>,
    continuation_scale_1: Param<4471>,
    null_move_reduction_gamma: Param<53773>,
    null_move_reduction_delta: Param<42808>,
    fail_high_reduction_gamma: Param<464542>,
    fail_high_reduction_delta: Param<233308>,
    fail_low_reduction_gamma: Param<1556236>,
    fail_low_reduction_delta: Param<353921>,
    single_extension_margin_gamma: Param<3116>,
    single_extension_margin_delta: Param<1259>,
    double_extension_margin_gamma: Param<4406>,
    double_extension_margin_delta: Param<1696>,
    triple_extension_margin_gamma: Param<3283>,
    triple_extension_margin_delta: Param<593471>,
    razoring_margin_gamma: Param<276805>,
    razoring_margin_delta: Param<161804>,
    reverse_futility_margin_gamma: Param<54077>,
    reverse_futility_margin_delta: Param<30370>,
    futility_margin_gamma: Param<159754>,
    futility_margin_delta: Param<271506>,
    futility_pruning_threshold_gamma: Param<37192>,
    see_pruning_threshold_gamma: Param<239747>,
    late_move_reduction_gamma: Param<1357>,
    late_move_reduction_delta: Param<2865>,
    late_move_reduction_pv: Param<4292>,
    late_move_reduction_cut: Param<4334>,
    late_move_reduction_improving: Param<1534>,
    late_move_reduction_history: Param<4380>,
    late_move_reduction_killer: Param<4223>,
    late_move_reduction_check: Param<4118>,
    late_move_reduction_tt_noisy: Param<4102>,
    late_move_pruning_gamma: Param<1933>,
    late_move_pruning_delta: Param<4853>,
    killer_move_bonus: Param<216731>,
    continuation_rating_scale_1: Param<5235>,
    noisy_gain_rating_gamma: Param<6129>,
    noisy_gain_rating_delta: Param<58527>,
    aspiration_window_start: Param<18582>,
    aspiration_window_gamma: Param<5461>,
    aspiration_window_delta: Param<6111>,
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

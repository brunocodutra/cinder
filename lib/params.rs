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
    moves_left_start: Param<706611>,
    moves_left_end: Param<9448>,
    soft_time_fraction: Param<2341>,
    hard_time_fraction: Param<3186>,
    score_trend_inertia: Param<25075>,
    pv_focus_gamma: Param<7775>,
    pv_focus_delta: Param<8906>,
    score_trend_magnitude: Param<1955>,
    score_trend_pivot: Param<234875>,
    quiet_history_bonus_gamma: Param<5642>,
    quiet_history_bonus_delta: Param<1005>,
    noisy_history_bonus_gamma: Param<6913>,
    noisy_history_bonus_delta: Param<1274>,
    quiet_continuation_bonus_gamma: Param<7033>,
    quiet_continuation_bonus_delta: Param<718>,
    noisy_continuation_bonus_gamma: Param<7168>,
    noisy_continuation_bonus_delta: Param<1174>,
    quiet_history_penalty_gamma: Param<7888>,
    quiet_history_penalty_delta: Param<1645>,
    noisy_history_penalty_gamma: Param<7372>,
    noisy_history_penalty_delta: Param<1731>,
    quiet_continuation_penalty_gamma: Param<6904>,
    quiet_continuation_penalty_delta: Param<1085>,
    noisy_continuation_penalty_gamma: Param<7494>,
    noisy_continuation_penalty_delta: Param<549>,
    continuation_scale_1: Param<4452>,
    null_move_reduction_gamma: Param<53691>,
    null_move_reduction_delta: Param<43832>,
    fail_high_reduction_gamma: Param<454417>,
    fail_high_reduction_delta: Param<231888>,
    fail_low_reduction_gamma: Param<1471558>,
    fail_low_reduction_delta: Param<381086>,
    single_extension_margin_gamma: Param<3258>,
    single_extension_margin_delta: Param<1049>,
    double_extension_margin_gamma: Param<4324>,
    double_extension_margin_delta: Param<1920>,
    triple_extension_margin_gamma: Param<3554>,
    triple_extension_margin_delta: Param<574971>,
    razoring_margin_gamma: Param<310525>,
    razoring_margin_delta: Param<163956>,
    reverse_futility_margin_gamma: Param<58240>,
    reverse_futility_margin_delta: Param<29549>,
    futility_margin_gamma: Param<158604>,
    futility_margin_delta: Param<293021>,
    futility_pruning_threshold_gamma: Param<36228>,
    see_pruning_threshold_gamma: Param<250564>,
    late_move_reduction_gamma: Param<1496>,
    late_move_reduction_delta: Param<2423>,
    late_move_reduction_pv: Param<4303>,
    late_move_reduction_cut: Param<4217>,
    late_move_reduction_improving: Param<2018>,
    late_move_reduction_history: Param<4601>,
    late_move_reduction_killer: Param<4389>,
    late_move_reduction_check: Param<3753>,
    late_move_reduction_tt_noisy: Param<4100>,
    late_move_pruning_gamma: Param<2215>,
    late_move_pruning_delta: Param<5527>,
    killer_move_bonus: Param<234438>,
    continuation_rating_scale_1: Param<4692>,
    noisy_gain_rating_gamma: Param<5877>,
    noisy_gain_rating_delta: Param<56534>,
    aspiration_window_start: Param<18527>,
    aspiration_window_gamma: Param<5637>,
    aspiration_window_delta: Param<5468>,
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

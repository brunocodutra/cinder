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
    moves_left_start: Param<684602>,
    moves_left_end: Param<8471>,
    soft_time_fraction: Param<2875>,
    hard_time_fraction: Param<3256>,
    score_trend_inertia: Param<28944>,
    pv_focus_gamma: Param<7021>,
    pv_focus_delta: Param<7840>,
    score_trend_magnitude: Param<3224>,
    score_trend_pivot: Param<172764>,
    pawn_correction: Param<54590>,
    minor_correction: Param<49843>,
    major_correction: Param<58395>,
    pieces_correction: Param<62439>,
    correction_gradient_gamma: Param<1175>,
    correction_gradient_delta: Param<2762>,
    correction_gradient_limit: Param<15876>,
    pawn_correction_bonus: Param<4433>,
    minor_correction_bonus: Param<4030>,
    major_correction_bonus: Param<3713>,
    pieces_correction_bonus: Param<3579>,
    quiet_history_bonus_gamma: Param<7894>,
    quiet_history_bonus_delta: Param<1597>,
    noisy_history_bonus_gamma: Param<5235>,
    noisy_history_bonus_delta: Param<1091>,
    quiet_continuation_bonus_gamma: Param<9275>,
    quiet_continuation_bonus_delta: Param<275>,
    noisy_continuation_bonus_gamma: Param<8104>,
    noisy_continuation_bonus_delta: Param<1908>,
    quiet_history_penalty_gamma: Param<9760>,
    quiet_history_penalty_delta: Param<3018>,
    noisy_history_penalty_gamma: Param<7727>,
    noisy_history_penalty_delta: Param<2420>,
    quiet_continuation_penalty_gamma: Param<9544>,
    quiet_continuation_penalty_delta: Param<707>,
    noisy_continuation_penalty_gamma: Param<8527>,
    noisy_continuation_penalty_delta: Param<969>,
    null_move_reduction_gamma: Param<53350>,
    null_move_reduction_delta: Param<36385>,
    fail_high_reduction_gamma: Param<418032>,
    fail_high_reduction_delta: Param<252225>,
    fail_low_reduction_gamma: Param<1606481>,
    fail_low_reduction_delta: Param<396207>,
    single_extension_margin_gamma: Param<2479>,
    single_extension_margin_delta: Param<2217>,
    double_extension_margin_gamma: Param<4692>,
    double_extension_margin_delta: Param<1185>,
    triple_extension_margin_gamma: Param<2492>,
    triple_extension_margin_delta: Param<583407>,
    razoring_margin_gamma: Param<265229>,
    razoring_margin_delta: Param<162464>,
    reverse_futility_margin_gamma: Param<61267>,
    reverse_futility_margin_delta: Param<31968>,
    futility_margin_gamma: Param<159416>,
    futility_margin_delta: Param<273979>,
    futility_pruning_threshold: Param<38362>,
    see_pruning_threshold: Param<226879>,
    late_move_reduction_gamma: Param<1071>,
    late_move_reduction_delta: Param<3501>,
    late_move_reduction_pv: Param<4470>,
    late_move_reduction_cut: Param<4996>,
    late_move_reduction_improving: Param<1508>,
    late_move_reduction_history: Param<4591>,
    late_move_reduction_continuation: Param<4901>,
    late_move_reduction_killer: Param<4564>,
    late_move_reduction_check: Param<3694>,
    late_move_reduction_tt_noisy: Param<4013>,
    late_move_pruning_gamma: Param<2090>,
    late_move_pruning_delta: Param<4795>,
    killer_move_bonus: Param<222618>,
    history_rating: Param<4513>,
    continuation_rating: Param<6230>,
    winning_rating_gamma: Param<5700>,
    winning_rating_delta: Param<58992>,
    aspiration_window_start: Param<16383>,
    aspiration_window_gamma: Param<5560>,
    aspiration_window_delta: Param<6458>,
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

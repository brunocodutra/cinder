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
    moves_left_start: Param<729101, 0>,
    moves_left_end: Param<8541, 0>,
    soft_time_fraction: Param<3055, 0>,
    hard_time_fraction: Param<3501, 0>,
    score_trend_inertia: Param<28907, 0>,
    pv_focus_gamma: Param<6752, 0>,
    pv_focus_delta: Param<7735, 0>,
    score_trend_magnitude: Param<3439, 0>,
    score_trend_pivot: Param<156366>,
    pawn_correction: Param<46450>,
    minor_correction: Param<46207>,
    major_correction: Param<61682>,
    pieces_correction: Param<64449>,
    correction_gradient_gamma: Param<1155>,
    correction_gradient_delta: Param<2410>,
    correction_gradient_limit: Param<14085>,
    pawn_correction_bonus: Param<4387>,
    minor_correction_bonus: Param<4473>,
    major_correction_bonus: Param<3829>,
    pieces_correction_bonus: Param<3775>,
    quiet_history_bonus_gamma: Param<8329>,
    quiet_history_bonus_delta: Param<1064>,
    noisy_history_bonus_gamma: Param<5457>,
    noisy_history_bonus_delta: Param<688>,
    quiet_continuation_bonus_gamma: Param<10108>,
    quiet_continuation_bonus_delta: Param<990>,
    noisy_continuation_bonus_gamma: Param<7914>,
    noisy_continuation_bonus_delta: Param<1466>,
    quiet_history_penalty_gamma: Param<9572>,
    quiet_history_penalty_delta: Param<2755>,
    noisy_history_penalty_gamma: Param<8611>,
    noisy_history_penalty_delta: Param<2010>,
    quiet_continuation_penalty_gamma: Param<8730>,
    quiet_continuation_penalty_delta: Param<506>,
    noisy_continuation_penalty_gamma: Param<8979>,
    noisy_continuation_penalty_delta: Param<1616>,
    null_move_reduction_gamma: Param<50716>,
    null_move_reduction_delta: Param<30505>,
    fail_high_reduction_gamma: Param<461871>,
    fail_high_reduction_delta: Param<230467>,
    fail_low_reduction_gamma: Param<1654252>,
    fail_low_reduction_delta: Param<414757>,
    improving_2: Param<4301>,
    improving_4: Param<4325>,
    single_extension_margin_gamma: Param<2375>,
    single_extension_margin_delta: Param<1857>,
    double_extension_margin_gamma: Param<4292>,
    double_extension_margin_delta: Param<1619>,
    triple_extension_margin_gamma: Param<3141>,
    triple_extension_margin_delta: Param<561447>,
    razoring_margin_gamma: Param<235438>,
    razoring_margin_delta: Param<163353>,
    reverse_futility_margin_gamma: Param<60319>,
    reverse_futility_margin_delta: Param<33069>,
    futility_margin_gamma: Param<136878>,
    futility_margin_delta: Param<276608>,
    futility_margin_is_pv: Param<28511>,
    futility_margin_was_pv: Param<27378>,
    futility_margin_gain: Param<4487>,
    futility_margin_killer: Param<27778>,
    futility_margin_check: Param<27651>,
    futility_margin_history: Param<19497>,
    futility_margin_continuation: Param<23512>,
    futility_margin_improving: Param<20000>,
    see_pruning_gamma: Param<250000>,
    see_pruning_delta: Param<4096>,
    see_pruning_killer: Param<30000>,
    see_pruning_history: Param<10000>,
    see_pruning_continuation: Param<20000>,
    late_move_reduction_gamma: Param<1094>,
    late_move_reduction_delta: Param<3876>,
    late_move_reduction_baseline: Param<986>,
    late_move_reduction_root: Param<2048>,
    late_move_reduction_is_pv: Param<4296>,
    late_move_reduction_was_pv: Param<995>,
    late_move_reduction_killer: Param<4430>,
    late_move_reduction_check: Param<3898>,
    late_move_reduction_history: Param<4297>,
    late_move_reduction_continuation: Param<4967>,
    late_move_reduction_improving: Param<1626>,
    late_move_reduction_cut: Param<4677>,
    late_move_reduction_noisy_pv: Param<3513>,
    late_move_pruning_gamma: Param<1773>,
    late_move_pruning_delta: Param<4355>,
    late_move_pruning_baseline: Param<4123>,
    late_move_pruning_root: Param<4096>,
    late_move_pruning_is_pv: Param<4096>,
    late_move_pruning_was_pv: Param<4096>,
    late_move_pruning_check: Param<4096>,
    late_move_pruning_improving: Param<4305>,
    killer_move_bonus: Param<224117>,
    history_rating: Param<3888>,
    continuation_rating: Param<5460>,
    winning_rating_gamma: Param<6074>,
    winning_rating_delta: Param<60289>,
    aspiration_window_start: Param<18281>,
    aspiration_window_gamma: Param<5515>,
    aspiration_window_delta: Param<6763>,
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

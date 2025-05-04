use crate::util::{Integer, Primitive};
use std::{cell::SyncUnsafeCell, mem::MaybeUninit};

#[cfg(not(feature = "spsa"))]
mod constant;

#[cfg(not(feature = "spsa"))]
pub use constant::*;

#[cfg(feature = "spsa")]
mod variable;

#[cfg(feature = "spsa")]
pub use variable::*;

impl<const VALUE: i32, const MIN: i32, const MAX: i32, const BASE: i32>
    Param<VALUE, MIN, MAX, BASE>
{
    const _VALID: () = const { assert!(MIN <= VALUE && VALUE <= MAX) };

    /// The parameter value as floating point.
    #[inline(always)]
    pub fn as_float(&self) -> f64 {
        self.get() as f64 / BASE as f64
    }

    /// The parameter value as integer.
    #[inline(always)]
    pub fn as_int<T: Primitive>(&self) -> T {
        (self.get() / BASE).cast()
    }
}

static PARAMS: SyncUnsafeCell<Params> = unsafe { MaybeUninit::zeroed().assume_init() };

#[cfg(feature = "spsa")]
macro_rules! len {
    ($name:ident,) => { 1 };
    ($first:ident, $($rest:ident,)*) => {
        1 + len!($($rest,)*)
    }
}

macro_rules! params {
    ($($name: ident: $type: ty,)*) => {
        #[cfg(feature = "spsa")]
        use serde::{Deserialize, Serialize};

        #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
        #[cfg_attr(test, derive(test_strategy::Arbitrary))]
        #[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
        #[cfg_attr(feature = "spsa", serde(deny_unknown_fields))]
        pub struct Params {
            $(#[cfg_attr(feature = "spsa", serde(default))] $name: $type,)*
        }

        $(impl Params {
            /// This parameter's current value.
            #[inline(always)]
            pub fn $name() -> &'static $type {
                unsafe { &PARAMS.get().as_ref_unchecked().$name }
            }
        })*

        #[cfg(feature = "spsa")]
        impl Params {
            // The number of parameters.
            pub const LEN: usize = len!($($name,)*);

            /// Perturb parameters in both positive and negative directions.
            ///
            /// # Panic
            ///
            /// Panics if the `perturbations`'s length is less than [`Self::LEN`].
            pub fn perturb<I: IntoIterator<Item = f64>>(&self, perturbations: I) -> (Self, Self) {
                use crate::util::{Assume, Bounded};

                let mut perturbations = perturbations.into_iter();
                let (mut left, mut right) = (Self::default(), Self::default());

                $(
                    let delta = self.$name.range() * perturbations.next().unwrap();
                    let value: Bounded<$type> = self.$name.convert().assume();
                    left.$name = (value + delta.round() as i32).convert().assume();
                    right.$name = (value - delta.round() as i32).convert().assume();
                )*

                (left, right)
            }

            /// Update parameters in-place.
            ///
            /// # Panic
            ///
            /// Panics if the `corrections`'s length is less than [`Self::LEN`].
            pub fn update<I: IntoIterator<Item = f64>>(&mut self, corrections: I) {
                use crate::util::{Assume, Bounded};

                let mut corrections = corrections.into_iter();

                $(
                    let delta = self.$name.range() * corrections.next().unwrap();
                    let value: Bounded<$type> = self.$name.convert().assume();
                    self.$name = (value + delta.round() as i32).convert().assume();
                )*

            }
        }
    };
}

params! {
    value_scale: Param<128, 128, 128, 1>,
    moves_left_start: Param<2577, 0, 4000, 10>,
    moves_left_end: Param<2286, 0, 4000, 1000>,
    soft_time_fraction: Param<6428, 3000, 10000, 10000>,
    hard_time_fraction: Param<4944, 2000, 8000, 10000>,
    score_trend_inertia: Param<6088, 3000, 10000, 1000>,
    pv_focus_alpha: Param<2290, 0, 4000, 1000>,
    pv_focus_beta: Param<2494, 0, 4000, 1000>,
    score_trend_magnitude: Param<4792, 2000, 8000, 10000>,
    score_trend_pivot: Param<6596, 3000, 10000, 1>,
    history_bonus_scale: Param<8192, 8192, 8192, 1>,
    history_bonus_alpha: Param<8336, 4000, 13000, 1>,
    history_bonus_beta: Param<111, 0, 4000, 1>,
    history_penalty_scale: Param<8192, 8192, 8192, 1>,
    history_penalty_alpha: Param<8392, 4000, 13000, 1>,
    history_penalty_beta: Param<363, 0, 4000, 1>,
    null_move_reduction_alpha: Param<1321, 0, 4000, 1>,
    null_move_reduction_beta: Param<1499, 0, 4000, 1>,
    fail_high_reduction_alpha: Param<15991, 7000, 24000, 1>,
    fail_high_reduction_beta: Param<6969, 3000, 11000, 1>,
    fail_low_reduction_alpha: Param<43286, 21000, 65000, 1>,
    fail_low_reduction_beta: Param<10805, 5000, 17000, 1>,
    singular_extension_margin_alpha: Param<11092, 5000, 17000, 100>,
    singular_extension_margin_beta: Param<2722, 0, 4000, 100>,
    futility_margin_alpha: Param<11272, 5000, 17000, 1>,
    futility_margin_beta: Param<7762, 3000, 12000, 1>,
    futility_pruning_threshold_alpha: Param<424, 0, 4000, 1>,
    see_pruning_threshold_alpha: Param<6772, 3000, 11000, 1>,
    late_move_reduction_scale: Param<8192, 8192, 8192, 1>,
    late_move_reduction_alpha: Param<4276, 2000, 7000, 1>,
    late_move_reduction_beta: Param<423, 0, 4000, 1>,
    late_move_pruning_scale: Param<8192, 8192, 8192, 1>,
    late_move_pruning_alpha: Param<5693, 2000, 9000, 1>,
    late_move_pruning_beta: Param<10389, 5000, 16000, 1>,
    killer_move_bonus: Param<12451, 6000, 19000, 1>,
    aspiration_window_start: Param<6046, 3000, 10000, 10>,
    aspiration_window_alpha: Param<1532, 0, 4000, 10>,
    aspiration_window_beta: Param<694, 0, 4000, 10>,
}

#[cfg(test)]
#[cfg(feature = "spsa")]
mod tests {
    use super::*;
    use proptest::sample::size_range;
    use test_strategy::proptest;

    #[proptest]
    fn perturbing_params_adjusts_by_at_least_grain(
        p: Params,
        #[any(size_range(Params::LEN).lift())] d: Vec<f64>,
    ) {
        let (mut l, mut r) = (p, p);
        l.update(d.iter().copied());
        r.update(d.iter().map(|&d| -d));
        assert_eq!(p.perturb(d), (l, r));
    }
}

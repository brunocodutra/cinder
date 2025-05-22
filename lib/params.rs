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
    moves_left_start: Param<2469, 0, 4000, 10>,
    moves_left_end: Param<2440, 0, 4000, 1000>,
    soft_time_fraction: Param<6169, 3000, 10000, 10000>,
    hard_time_fraction: Param<5271, 2000, 8000, 10000>,
    score_trend_inertia: Param<5697, 3000, 10000, 1000>,
    pv_focus_alpha: Param<2012, 0, 4000, 1000>,
    pv_focus_beta: Param<2394, 0, 4000, 1000>,
    score_trend_magnitude: Param<4581, 2000, 8000, 10000>,
    score_trend_pivot: Param<6741, 3000, 10000, 1>,
    history_bonus_scale: Param<8192, 8192, 8192, 1>,
    history_bonus_alpha: Param<9419, 4000, 13000, 1>,
    history_bonus_beta: Param<69, 0, 4000, 1>,
    history_penalty_scale: Param<8192, 8192, 8192, 1>,
    history_penalty_alpha: Param<9075, 4000, 13000, 1>,
    history_penalty_beta: Param<521, 0, 4000, 1>,
    null_move_reduction_alpha: Param<1499, 0, 4000, 1>,
    null_move_reduction_beta: Param<1609, 0, 4000, 1>,
    fail_high_reduction_alpha: Param<15690, 7000, 24000, 1>,
    fail_high_reduction_beta: Param<7613, 3000, 11000, 1>,
    fail_low_reduction_alpha: Param<45402, 21000, 65000, 1>,
    fail_low_reduction_beta: Param<10992, 5000, 17000, 1>,
    singular_extension_margin_alpha: Param<10465, 5000, 17000, 100>,
    singular_extension_margin_beta: Param<2680, 0, 4000, 100>,
    double_extension_margin_alpha: Param<12800, 5000, 17000, 100>,
    double_extension_margin_beta: Param<5000, 0, 8000, 100>,
    futility_margin_alpha: Param<9815, 5000, 17000, 1>,
    futility_margin_beta: Param<7821, 3000, 12000, 1>,
    futility_pruning_threshold_alpha: Param<682, 0, 4000, 1>,
    see_pruning_threshold_alpha: Param<7272, 3000, 11000, 1>,
    late_move_reduction_scale: Param<8192, 8192, 8192, 1>,
    late_move_reduction_alpha: Param<4231, 2000, 7000, 1>,
    late_move_reduction_beta: Param<727, 0, 4000, 1>,
    late_move_pruning_scale: Param<8192, 8192, 8192, 1>,
    late_move_pruning_alpha: Param<5094, 2000, 9000, 1>,
    late_move_pruning_beta: Param<9783, 5000, 16000, 1>,
    killer_move_bonus: Param<11528, 6000, 19000, 1>,
    aspiration_window_start: Param<6215, 3000, 10000, 10>,
    aspiration_window_alpha: Param<1538, 0, 4000, 10>,
    aspiration_window_beta: Param<1398, 0, 4000, 10>,
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

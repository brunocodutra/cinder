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
    moves_left_start: Param<22500, 10000, 40000, 100>,
    moves_left_end: Param<300, 100, 1000, 100>,
    soft_time_fraction: Param<50, 10, 100, 100>,
    hard_time_fraction: Param<50, 50, 100, 100>,
    score_trend_inertia: Param<700, 100, 1400, 100>,
    pv_focus_alpha: Param<280, 100, 500, 100>,
    pv_focus_beta: Param<300, 100, 500, 100>,
    score_trend_magnitude: Param<50, 10, 90, 100>,
    score_trend_pivot: Param<6400, 4000, 8000, 1>,
    history_bonus_scale: Param<8192, 8192, 8192, 1>,
    history_bonus_alpha: Param<8192, 6000, 10000, 1>,
    history_bonus_beta: Param<0, 0, 10000, 1>,
    history_penalty_scale: Param<8192, 8192, 8192, 1>,
    history_penalty_alpha: Param<8192, 6000, 10000, 1>,
    history_penalty_beta: Param<0, 0, 10000, 1>,
    null_move_reduction_alpha: Param<1280, 1000, 4000, 1>,
    null_move_reduction_beta: Param<1280, 1000, 4000, 1>,
    fail_high_reduction_alpha: Param<17920, 5000, 20000, 1>,
    fail_high_reduction_beta: Param<7680, 5000, 20000, 1>,
    fail_low_reduction_alpha: Param<46080, 20000, 60000, 1>,
    fail_low_reduction_beta: Param<23040, 20000, 60000, 1>,
    singular_extension_margin_alpha: Param<128, 50, 200, 1>,
    singular_extension_margin_beta: Param<0, 0, 200, 1>,
    futility_margin_alpha: Param<10240, 6000, 12000, 1>,
    futility_margin_beta: Param<7680, 6000, 12000, 1>,
    futility_pruning_threshold_alpha: Param<0, 0, 20000, 1>,
    see_pruning_threshold_alpha: Param<9600, 0, 20000, 1>,
    late_move_reduction_scale: Param<8192, 8192, 8192, 1>,
    late_move_reduction_alpha: Param<4096, 2000, 8000, 1>,
    late_move_reduction_beta: Param<0, 0, 8000, 1>,
    late_move_pruning_scale: Param<8192, 8192, 8192, 1>,
    late_move_pruning_alpha: Param<4096, 3000, 15000, 1>,
    late_move_pruning_beta: Param<12288, 3000, 15000, 1>,
    killer_move_bonus: Param<16384, 10000, 40000, 1>,
    aspiration_window_start: Param<640, 200, 800, 1>,
    aspiration_window_alpha: Param<256, 200, 800, 1>,
    aspiration_window_beta: Param<0, 0, 800, 1>,
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

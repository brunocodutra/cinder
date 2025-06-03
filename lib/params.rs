use crate::util::{Integer, Primitive};
use std::cell::SyncUnsafeCell;

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

static PARAMS: SyncUnsafeCell<Params> = SyncUnsafeCell::new(Params::new());

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

        impl Params {
            const fn new() -> Self {
                Params {
                    $($name: <$type>::new(),)*
                }
            }
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
    moves_left_start: Param<2244, 0, 4000, 10>,
    moves_left_end: Param<2155, 0, 4000, 1000>,
    soft_time_fraction: Param<5998, 2000, 9000, 10000>,
    hard_time_fraction: Param<5895, 2000, 9000, 10000>,
    score_trend_inertia: Param<5184, 2000, 8000, 1000>,
    pv_focus_alpha: Param<2113, 0, 4000, 1000>,
    pv_focus_beta: Param<2491, 0, 4000, 1000>,
    score_trend_magnitude: Param<4886, 2000, 8000, 10000>,
    score_trend_pivot: Param<7212, 3000, 11000, 1>,
    history_bonus_scale: Param<8192, 8192, 8192, 1>,
    history_bonus_alpha: Param<10873, 5000, 17000, 1>,
    history_bonus_beta: Param<2106, 0, 4000, 1>,
    history_penalty_scale: Param<8192, 8192, 8192, 1>,
    history_penalty_alpha: Param<9770, 4000, 15000, 1>,
    history_penalty_beta: Param<2134, 0, 4000, 1>,
    null_move_reduction_alpha: Param<1939, 0, 4000, 1>,
    null_move_reduction_beta: Param<1817, 0, 4000, 1>,
    fail_high_reduction_alpha: Param<14198, 7000, 22000, 1>,
    fail_high_reduction_beta: Param<7977, 3000, 12000, 1>,
    fail_low_reduction_alpha: Param<41909, 20000, 63000, 1>,
    fail_low_reduction_beta: Param<11390, 5000, 18000, 1>,
    single_extension_margin_alpha: Param<10427, 5000, 16000, 100>,
    single_extension_margin_beta: Param<3002, 1000, 5000, 100>,
    double_extension_margin_alpha: Param<12020, 6000, 19000, 100>,
    double_extension_margin_beta: Param<4963, 2000, 8000, 100>,
    reverse_futility_margin_alpha: Param<2048, 0, 4000, 1>,
    reverse_futility_margin_beta: Param<1000, 0, 4000, 1>,
    futility_margin_alpha: Param<6597, 3000, 10000, 1>,
    futility_margin_beta: Param<8361, 4000, 13000, 1>,
    futility_pruning_threshold_alpha: Param<1042, 0, 4000, 1>,
    see_pruning_threshold_alpha: Param<7129, 3000, 11000, 1>,
    late_move_reduction_scale: Param<8192, 8192, 8192, 1>,
    late_move_reduction_alpha: Param<4212, 2000, 7000, 1>,
    late_move_reduction_beta: Param<1011, 0, 4000, 1>,
    late_move_pruning_scale: Param<8192, 8192, 8192, 1>,
    late_move_pruning_alpha: Param<5270, 2000, 8000, 1>,
    late_move_pruning_beta: Param<9541, 4000, 15000, 1>,
    killer_move_bonus: Param<9690, 4000, 15000, 1>,
    noisy_gain_rating_alpha: Param<2024, 0, 4000, 10>,
    noisy_gain_rating_beta: Param<19146, 9000, 29000, 10>,
    aspiration_window_start: Param<6200, 3000, 10000, 10>,
    aspiration_window_alpha: Param<1716, 0, 4000, 10>,
    aspiration_window_beta: Param<1740, 0, 4000, 10>,
}

#[cfg(test)]
#[cfg(feature = "spsa")]
mod tests {
    use super::*;
    use proptest::sample::size_range;
    use test_strategy::proptest;

    #[proptest]
    fn perturbing_updates_params(p: Params, #[any(size_range(Params::LEN).lift())] d: Vec<f64>) {
        let (mut l, mut r) = (p, p);
        l.update(d.iter().copied());
        r.update(d.iter().map(|&d| -d));
        assert_eq!(p.perturb(d), (l, r));
    }
}

use std::cell::SyncUnsafeCell;

#[cfg(not(feature = "spsa"))]
mod constant;

#[cfg(not(feature = "spsa"))]
pub use constant::*;

#[cfg(feature = "spsa")]
mod variable;

#[cfg(feature = "spsa")]
pub use variable::*;

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
                #[cfg(feature = "spsa")]
                use crate::util::Integer;

                unsafe { PARAMS.get().as_ref_unchecked().$name.get() }
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
                use crate::util::{Assume, Bounded, Integer};

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
                use crate::util::{Assume, Bounded, Integer};

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
    value_scale: Param<524288, 524288, 524288>,
    moves_left_start: Param<919142, 459000, 1379000>,
    moves_left_end: Param<8827, 4000, 14000>,
    soft_time_fraction: Param<2457, 0, 4000>,
    hard_time_fraction: Param<2415, 0, 4000>,
    score_trend_inertia: Param<21234, 10000, 32000>,
    pv_focus_gamma: Param<8655, 4000, 13000>,
    pv_focus_delta: Param<10203, 5000, 16000>,
    score_trend_magnitude: Param<2001, 0, 4000>,
    score_trend_pivot: Param<230784, 115000, 347000>,
    history_bonus_gamma: Param<5437, 2000, 9000>,
    history_bonus_delta: Param<1053, 0, 4000>,
    history_penalty_gamma: Param<4885, 2000, 8000>,
    history_penalty_delta: Param<1067, 0, 4000>,
    null_move_reduction_gamma: Param<62048, 31000, 94000>,
    null_move_reduction_delta: Param<58144, 29000, 88000>,
    fail_high_reduction_gamma: Param<454336, 227000, 682000>,
    fail_high_reduction_delta: Param<255264, 127000, 383000>,
    fail_low_reduction_gamma: Param<1341088, 670000, 2012000>,
    fail_low_reduction_delta: Param<364480, 182000, 547000>,
    single_extension_margin_gamma: Param<3337, 1000, 6000>,
    single_extension_margin_delta: Param<961, 0, 4000>,
    double_extension_margin_gamma: Param<3846, 1000, 6000>,
    double_extension_margin_delta: Param<1588, 0, 4000>,
    razoring_margin_gamma: Param<320000, 160000, 480000>,
    razoring_margin_delta: Param<160000, 80000, 240000>,
    reverse_futility_margin_gamma: Param<65536, 32000, 99000>,
    reverse_futility_margin_delta: Param<32000, 16000, 48000>,
    futility_margin_gamma: Param<211104, 105000, 317000>,
    futility_margin_delta: Param<267552, 133000, 402000>,
    futility_pruning_threshold_gamma: Param<33344, 16000, 51000>,
    see_pruning_threshold_gamma: Param<228128, 114000, 343000>,
    late_move_reduction_gamma: Param<2106, 0, 4000>,
    late_move_reduction_delta: Param<506, 0, 4000>,
    late_move_pruning_gamma: Param<2635, 0, 4000>,
    late_move_pruning_delta: Param<4771, 2000, 8000>,
    killer_move_bonus: Param<310080, 155000, 466000>,
    noisy_gain_rating_gamma: Param<6477, 3000, 10000>,
    noisy_gain_rating_delta: Param<61267, 30000, 92000>,
    aspiration_window_start: Param<19840, 9000, 30000>,
    aspiration_window_gamma: Param<5491, 2000, 9000>,
    aspiration_window_delta: Param<5568, 2000, 9000>,
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

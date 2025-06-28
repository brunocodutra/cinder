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
    score_trend_pivot: Param<165181>,
    pawn_correction: Param<48368>,
    minor_correction: Param<44079>,
    major_correction: Param<59437>,
    pieces_correction: Param<63938>,
    correction_gradient_gamma: Param<1118>,
    correction_gradient_delta: Param<2538>,
    correction_gradient_limit: Param<13840>,
    pawn_correction_bonus: Param<4411>,
    minor_correction_bonus: Param<4414>,
    major_correction_bonus: Param<3871>,
    pieces_correction_bonus: Param<3733>,
    quiet_history_bonus_gamma: Param<8204>,
    quiet_history_bonus_delta: Param<876>,
    noisy_history_bonus_gamma: Param<5480>,
    noisy_history_bonus_delta: Param<782>,
    quiet_continuation_bonus_gamma: Param<10286>,
    quiet_continuation_bonus_delta: Param<1004>,
    noisy_continuation_bonus_gamma: Param<8132>,
    noisy_continuation_bonus_delta: Param<1523>,
    quiet_history_penalty_gamma: Param<9185>,
    quiet_history_penalty_delta: Param<2805>,
    noisy_history_penalty_gamma: Param<8717>,
    noisy_history_penalty_delta: Param<2037>,
    quiet_continuation_penalty_gamma: Param<8565>,
    quiet_continuation_penalty_delta: Param<512>,
    noisy_continuation_penalty_gamma: Param<8982>,
    noisy_continuation_penalty_delta: Param<1471>,
    null_move_reduction_gamma: Param<52027>,
    null_move_reduction_delta: Param<31166>,
    fail_high_reduction_gamma: Param<448012>,
    fail_high_reduction_delta: Param<232680>,
    fail_low_reduction_gamma: Param<1664500>,
    fail_low_reduction_delta: Param<402387>,
    improving_2: Param<4351>,
    improving_4: Param<4439>,
    single_extension_margin_gamma: Param<2479>,
    single_extension_margin_delta: Param<1890>,
    double_extension_margin_gamma: Param<4295>,
    double_extension_margin_delta: Param<1595>,
    triple_extension_margin_gamma: Param<3141>,
    triple_extension_margin_delta: Param<576680>,
    razoring_margin_gamma: Param<236456>,
    razoring_margin_delta: Param<162619>,
    reverse_futility_margin_gamma: Param<61225>,
    reverse_futility_margin_delta: Param<32443>,
    futility_margin_gamma: Param<139923>,
    futility_margin_delta: Param<274998>,
    futility_margin_gain: Param<4554>,
    futility_margin_killer: Param<28621>,
    futility_margin_check: Param<28123>,
    futility_margin_history: Param<19272>,
    futility_margin_continuation: Param<24463>,
    futility_margin_is_pv: Param<27897>,
    futility_margin_was_pv: Param<27737>,
    see_pruning_threshold: Param<233461>,
    late_move_reduction_gamma: Param<1170>,
    late_move_reduction_delta: Param<3879>,
    late_move_reduction_baseline: Param<961>,
    late_move_reduction_is_pv: Param<4356>,
    late_move_reduction_was_pv: Param<928>,
    late_move_reduction_cut: Param<4793>,
    late_move_reduction_improving: Param<1649>,
    late_move_reduction_history: Param<4255>,
    late_move_reduction_continuation: Param<4963>,
    late_move_reduction_killer: Param<4426>,
    late_move_reduction_check: Param<3912>,
    late_move_reduction_tt_noisy: Param<3675>,
    late_move_pruning_gamma: Param<1910>,
    late_move_pruning_delta: Param<4371>,
    late_move_pruning_baseline: Param<4120>,
    late_move_pruning_improving: Param<4274>,
    killer_move_bonus: Param<238411>,
    history_rating: Param<4075>,
    continuation_rating: Param<5293>,
    winning_rating_gamma: Param<5988>,
    winning_rating_delta: Param<62817>,
    aspiration_window_start: Param<18595>,
    aspiration_window_gamma: Param<5271>,
    aspiration_window_delta: Param<6670>,
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

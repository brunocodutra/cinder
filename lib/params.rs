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
    score_trend_pivot: Param<195606>,
    pawn_correction: Param<61582>,
    minor_correction: Param<59893>,
    major_correction: Param<60714>,
    pieces_correction: Param<67367>,
    correction_gradient_gamma: Param<1185>,
    correction_gradient_delta: Param<2468>,
    correction_gradient_limit: Param<15977>,
    pawn_correction_bonus: Param<3926>,
    minor_correction_bonus: Param<3961>,
    major_correction_bonus: Param<3974>,
    pieces_correction_bonus: Param<3770>,
    quiet_history_bonus_gamma: Param<7017>,
    quiet_history_bonus_delta: Param<1675>,
    noisy_history_bonus_gamma: Param<5865>,
    noisy_history_bonus_delta: Param<1261>,
    quiet_continuation_bonus_gamma: Param<8575>,
    quiet_continuation_bonus_delta: Param<177>,
    noisy_continuation_bonus_gamma: Param<8781>,
    noisy_continuation_bonus_delta: Param<1501>,
    quiet_history_penalty_gamma: Param<9295>,
    quiet_history_penalty_delta: Param<2511>,
    noisy_history_penalty_gamma: Param<6945>,
    noisy_history_penalty_delta: Param<2165>,
    quiet_continuation_penalty_gamma: Param<9140>,
    quiet_continuation_penalty_delta: Param<1106>,
    noisy_continuation_penalty_gamma: Param<7528>,
    noisy_continuation_penalty_delta: Param<1120>,
    null_move_reduction_gamma: Param<57653>,
    null_move_reduction_delta: Param<35945>,
    fail_high_reduction_gamma: Param<402916>,
    fail_high_reduction_delta: Param<220870>,
    fail_low_reduction_gamma: Param<1570755>,
    fail_low_reduction_delta: Param<367795>,
    single_extension_margin_gamma: Param<2860>,
    single_extension_margin_delta: Param<1727>,
    double_extension_margin_gamma: Param<4447>,
    double_extension_margin_delta: Param<1332>,
    triple_extension_margin_gamma: Param<2912>,
    triple_extension_margin_delta: Param<582728>,
    razoring_margin_gamma: Param<260468>,
    razoring_margin_delta: Param<164300>,
    reverse_futility_margin_gamma: Param<56695>,
    reverse_futility_margin_delta: Param<31008>,
    futility_margin_gamma: Param<151458>,
    futility_margin_delta: Param<262550>,
    futility_pruning_threshold: Param<41456>,
    see_pruning_threshold: Param<241029>,
    late_move_reduction_gamma: Param<1020>,
    late_move_reduction_delta: Param<3114>,
    late_move_reduction_pv: Param<4064>,
    late_move_reduction_cut: Param<4962>,
    late_move_reduction_improving: Param<1226>,
    late_move_reduction_history: Param<5057>,
    late_move_reduction_continuation: Param<4480>,
    late_move_reduction_killer: Param<4142>,
    late_move_reduction_check: Param<3838>,
    late_move_reduction_tt_noisy: Param<3994>,
    late_move_pruning_gamma: Param<2195>,
    late_move_pruning_delta: Param<4802>,
    killer_move_bonus: Param<225723>,
    history_rating: Param<4594>,
    continuation_rating: Param<5977>,
    winning_rating_gamma: Param<6079>,
    winning_rating_delta: Param<59589>,
    aspiration_window_start: Param<18611>,
    aspiration_window_gamma: Param<4981>,
    aspiration_window_delta: Param<6323>,
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

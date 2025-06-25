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
    score_trend_pivot: Param<170209>,
    pawn_correction: Param<55463>,
    minor_correction: Param<44849>,
    major_correction: Param<59125>,
    pieces_correction: Param<65074>,
    correction_gradient_gamma: Param<1316>,
    correction_gradient_delta: Param<2934>,
    correction_gradient_limit: Param<15433>,
    pawn_correction_bonus: Param<4396>,
    minor_correction_bonus: Param<4082>,
    major_correction_bonus: Param<3803>,
    pieces_correction_bonus: Param<3620>,
    quiet_history_bonus_gamma: Param<8127>,
    quiet_history_bonus_delta: Param<1317>,
    noisy_history_bonus_gamma: Param<5122>,
    noisy_history_bonus_delta: Param<844>,
    quiet_continuation_bonus_gamma: Param<9736>,
    quiet_continuation_bonus_delta: Param<589>,
    noisy_continuation_bonus_gamma: Param<8422>,
    noisy_continuation_bonus_delta: Param<1666>,
    quiet_history_penalty_gamma: Param<9394>,
    quiet_history_penalty_delta: Param<2912>,
    noisy_history_penalty_gamma: Param<8195>,
    noisy_history_penalty_delta: Param<2378>,
    quiet_continuation_penalty_gamma: Param<8976>,
    quiet_continuation_penalty_delta: Param<487>,
    noisy_continuation_penalty_gamma: Param<8771>,
    noisy_continuation_penalty_delta: Param<1105>,
    null_move_reduction_gamma: Param<56326>,
    null_move_reduction_delta: Param<32541>,
    fail_high_reduction_gamma: Param<436225>,
    fail_high_reduction_delta: Param<267872>,
    fail_low_reduction_gamma: Param<1570417>,
    fail_low_reduction_delta: Param<383455>,
    single_extension_margin_gamma: Param<2545>,
    single_extension_margin_delta: Param<2300>,
    double_extension_margin_gamma: Param<4533>,
    double_extension_margin_delta: Param<1466>,
    triple_extension_margin_gamma: Param<2703>,
    triple_extension_margin_delta: Param<569392>,
    razoring_margin_gamma: Param<250631>,
    razoring_margin_delta: Param<169044>,
    reverse_futility_margin_gamma: Param<63818>,
    reverse_futility_margin_delta: Param<33054>,
    futility_margin_gamma: Param<150683>,
    futility_margin_delta: Param<277983>,
    futility_pruning_threshold: Param<40609>,
    see_pruning_threshold: Param<218534>,
    late_move_reduction_gamma: Param<1056>,
    late_move_reduction_delta: Param<3916>,
    late_move_reduction_pv: Param<4342>,
    late_move_reduction_cut: Param<5143>,
    late_move_reduction_improving: Param<1635>,
    late_move_reduction_history: Param<4173>,
    late_move_reduction_continuation: Param<4829>,
    late_move_reduction_killer: Param<4264>,
    late_move_reduction_check: Param<3651>,
    late_move_reduction_tt_noisy: Param<3972>,
    late_move_pruning_gamma: Param<1634>,
    late_move_pruning_delta: Param<4470>,
    killer_move_bonus: Param<233373>,
    history_rating: Param<4681>,
    continuation_rating: Param<5930>,
    winning_rating_gamma: Param<5904>,
    winning_rating_delta: Param<61354>,
    aspiration_window_start: Param<17132>,
    aspiration_window_gamma: Param<5469>,
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

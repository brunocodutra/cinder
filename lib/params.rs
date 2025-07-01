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
    score_trend_pivot: Param<168172>,
    pawns_correction: Param<47546>,
    minor_correction: Param<40148>,
    major_correction: Param<61497>,
    pieces_correction: Param<63836>,
    correction_gradient_gamma: Param<2080>,
    correction_gradient_delta: Param<3030>,
    pawns_correction_bonus: Param<3891>,
    minor_correction_bonus: Param<4956>,
    major_correction_bonus: Param<3466>,
    pieces_correction_bonus: Param<3506>,
    quiet_history_bonus_gamma: Param<8129>,
    quiet_history_bonus_delta: Param<1534>,
    noisy_history_bonus_gamma: Param<5778>,
    noisy_history_bonus_delta: Param<494>,
    quiet_continuation_bonus_gamma: Param<10842>,
    quiet_continuation_bonus_delta: Param<1634>,
    noisy_continuation_bonus_gamma: Param<9126>,
    noisy_continuation_bonus_delta: Param<1664>,
    quiet_history_penalty_gamma: Param<10793>,
    quiet_history_penalty_delta: Param<2986>,
    noisy_history_penalty_gamma: Param<8996>,
    noisy_history_penalty_delta: Param<2164>,
    quiet_continuation_penalty_gamma: Param<9843>,
    quiet_continuation_penalty_delta: Param<1291>,
    noisy_continuation_penalty_gamma: Param<8678>,
    noisy_continuation_penalty_delta: Param<1901>,
    null_move_reduction_gamma: Param<47191>,
    null_move_reduction_delta: Param<29295>,
    fail_high_reduction_gamma: Param<436519>,
    fail_high_reduction_delta: Param<229616>,
    fail_low_reduction_gamma: Param<1681659>,
    fail_low_reduction_delta: Param<449617>,
    improving_2: Param<4047>,
    improving_4: Param<4447>,
    single_extension_margin_gamma: Param<2419>,
    single_extension_margin_delta: Param<2120>,
    double_extension_margin_gamma: Param<4890>,
    double_extension_margin_delta: Param<935>,
    triple_extension_margin_gamma: Param<2380>,
    triple_extension_margin_delta: Param<557970>,
    razoring_margin_theta: Param<39610>,
    razoring_margin_gamma: Param<126528>,
    razoring_margin_delta: Param<154463>,
    reverse_futility_margin_theta: Param<8473>,
    reverse_futility_margin_gamma: Param<36246>,
    reverse_futility_margin_delta: Param<33322>,
    futility_margin_theta: Param<1995>,
    futility_margin_gamma: Param<157767>,
    futility_margin_delta: Param<241478>,
    futility_margin_is_pv: Param<28416>,
    futility_margin_was_pv: Param<28869>,
    futility_margin_gain: Param<5268>,
    futility_margin_killer: Param<29185>,
    futility_margin_check: Param<27802>,
    futility_margin_history: Param<18690>,
    futility_margin_continuation: Param<22330>,
    futility_margin_improving: Param<15022>,
    see_pruning_theta: Param<1930>,
    see_pruning_gamma: Param<271946>,
    see_pruning_delta: Param<3587>,
    see_pruning_killer: Param<27304>,
    see_pruning_history: Param<10063>,
    see_pruning_continuation: Param<20545>,
    late_move_reduction_theta: Param<800>,
    late_move_reduction_gamma: Param<400>,
    late_move_reduction_delta: Param<3081>,
    late_move_reduction_baseline: Param<1230>,
    late_move_reduction_root: Param<2210>,
    late_move_reduction_is_pv: Param<4200>,
    late_move_reduction_was_pv: Param<1290>,
    late_move_reduction_killer: Param<4899>,
    late_move_reduction_check: Param<3362>,
    late_move_reduction_history: Param<4203>,
    late_move_reduction_continuation: Param<4886>,
    late_move_reduction_improving: Param<1806>,
    late_move_reduction_cut: Param<4644>,
    late_move_reduction_noisy_pv: Param<3348>,
    late_move_pruning_theta: Param<1659>,
    late_move_pruning_gamma: Param<996>,
    late_move_pruning_delta: Param<4372>,
    late_move_pruning_baseline: Param<3253>,
    late_move_pruning_root: Param<4712>,
    late_move_pruning_is_pv: Param<3906>,
    late_move_pruning_was_pv: Param<3530>,
    late_move_pruning_check: Param<3543>,
    late_move_pruning_improving: Param<5185>,
    killer_move_bonus: Param<238373>,
    history_rating: Param<3553>,
    continuation_rating: Param<5689>,
    winning_rating_gamma: Param<6475>,
    winning_rating_delta: Param<53741>,
    aspiration_window_start: Param<19739>,
    aspiration_window_gamma: Param<6419>,
    aspiration_window_delta: Param<7046>,
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

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
    score_trend_pivot: Param<145600>,
    pawns_correction: Param<50316>,
    minor_correction: Param<44417>,
    major_correction: Param<52317>,
    pieces_correction: Param<61857>,
    correction_gradient_gamma: Param<2022>,
    correction_gradient_delta: Param<2808>,
    pawns_correction_bonus: Param<105628>,
    minor_correction_bonus: Param<119033>,
    major_correction_bonus: Param<124613>,
    pieces_correction_bonus: Param<113857>,
    history_bonus_quiet_gamma: Param<308419>,
    history_bonus_quiet_delta: Param<46496>,
    history_bonus_noisy_gamma: Param<204578>,
    history_bonus_noisy_delta: Param<10554>,
    history_penalty_quiet_gamma: Param<344062>,
    history_penalty_quiet_delta: Param<116351>,
    history_penalty_noisy_gamma: Param<304567>,
    history_penalty_noisy_delta: Param<84794>,
    continuation_bonus_quiet_gamma: Param<11481>,
    continuation_bonus_quiet_delta: Param<1928>,
    continuation_bonus_noisy_gamma: Param<9124>,
    continuation_bonus_noisy_delta: Param<1476>,
    continuation_penalty_quiet_gamma: Param<10104>,
    continuation_penalty_quiet_delta: Param<1341>,
    continuation_penalty_noisy_gamma: Param<8703>,
    continuation_penalty_noisy_delta: Param<1955>,
    improving_2: Param<3972>,
    improving_4: Param<4228>,
    fail_high_reduction_gamma: Param<424934>,
    fail_high_reduction_delta: Param<226015>,
    fail_low_reduction_gamma: Param<1552928>,
    fail_low_reduction_delta: Param<490660>,
    single_extension_margin_gamma: Param<2767>,
    single_extension_margin_delta: Param<1904>,
    double_extension_margin_gamma: Param<4959>,
    double_extension_margin_delta: Param<1187>,
    triple_extension_margin_gamma: Param<2470>,
    triple_extension_margin_delta: Param<620570>,
    null_move_pruning_gamma: Param<49218>,
    null_move_pruning_delta: Param<29358>,
    razoring_margin_theta: Param<42901>,
    razoring_margin_gamma: Param<98335>,
    razoring_margin_delta: Param<148767>,
    reverse_futility_margin_theta: Param<7341>,
    reverse_futility_margin_gamma: Param<45413>,
    reverse_futility_margin_delta: Param<34873>,
    reverse_futility_margin_improving: Param<20661>,
    reverse_futility_margin_noisy_pv: Param<4845>,
    reverse_futility_margin_cut: Param<10733>,
    futility_margin_theta: Param<1541>,
    futility_margin_gamma: Param<133928>,
    futility_margin_delta: Param<232438>,
    futility_margin_is_pv: Param<30176>,
    futility_margin_was_pv: Param<26686>,
    futility_margin_gain: Param<5710>,
    futility_margin_killer: Param<27863>,
    futility_margin_check: Param<27551>,
    futility_margin_history: Param<18111>,
    futility_margin_continuation: Param<20278>,
    futility_margin_improving: Param<16346>,
    see_pruning_theta: Param<2392>,
    see_pruning_gamma: Param<249649>,
    see_pruning_delta: Param<3922>,
    see_pruning_killer: Param<31273>,
    see_pruning_history: Param<10874>,
    see_pruning_continuation: Param<21552>,
    late_move_reduction_theta: Param<918>,
    late_move_reduction_gamma: Param<398>,
    late_move_reduction_delta: Param<2919>,
    late_move_reduction_baseline: Param<1219>,
    late_move_reduction_root: Param<2384>,
    late_move_reduction_is_pv: Param<2906>,
    late_move_reduction_was_pv: Param<901>,
    late_move_reduction_killer: Param<4802>,
    late_move_reduction_check: Param<3988>,
    late_move_reduction_history: Param<4127>,
    late_move_reduction_continuation: Param<5773>,
    late_move_reduction_improving: Param<1763>,
    late_move_reduction_noisy_pv: Param<3242>,
    late_move_reduction_cut: Param<5715>,
    late_move_pruning_theta: Param<1772>,
    late_move_pruning_gamma: Param<1160>,
    late_move_pruning_delta: Param<4131>,
    late_move_pruning_baseline: Param<3056>,
    late_move_pruning_root: Param<4422>,
    late_move_pruning_is_pv: Param<4235>,
    late_move_pruning_was_pv: Param<3281>,
    late_move_pruning_check: Param<3917>,
    late_move_pruning_improving: Param<4657>,
    killer_move_bonus: Param<226174>,
    history_rating: Param<467888>,
    continuation_rating: Param<569236>,
    winning_rating_gamma: Param<6712>,
    winning_rating_delta: Param<66957>,
    aspiration_window_start: Param<18196>,
    aspiration_window_gamma: Param<5288>,
    aspiration_window_delta: Param<7542>,
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

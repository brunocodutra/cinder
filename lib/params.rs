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

pub trait Param: Sized {
    #[cfg(feature = "spsa")]
    const LEN: usize;

    type Value;

    fn get(&self) -> Self::Value;

    #[cfg(feature = "spsa")]
    fn min() -> Self::Value;

    #[cfg(feature = "spsa")]
    fn max() -> Self::Value;

    #[cfg(feature = "spsa")]
    fn perturb<I: IntoIterator<Item = f64>>(&self, perturbations: I) -> (Self, Self);

    #[cfg(feature = "spsa")]
    fn update<I: IntoIterator<Item = f64>>(&mut self, corrections: I);
}

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
    ($name:ty,) => { 1 };
    ($first:ty, $($rest:ty,)*) => {
        <$first>::LEN + len!($($rest,)*)
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
            pub fn $name() -> <$type as Param>::Value {
                unsafe { PARAMS.get().as_ref_unchecked().$name.get() }
            }
        })*

        #[cfg(feature = "spsa")]
        impl Params {
            /// The number of parameters.
            pub const LEN: usize = len!($($type,)*);

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
    value_scale: Constant<524288>,
    moves_left_start: Constant<729101>,
    moves_left_end: Constant<8541>,
    soft_time_fraction: Constant<3055>,
    hard_time_fraction: Constant<3501>,
    score_trend_inertia: Constant<28907>,
    pv_focus_gamma: Constant<6752>,
    pv_focus_delta: Constant<7735>,
    score_trend_magnitude: Constant<3439>,
    score_trend_pivot: Scalar<128037>,
    pawns_correction: Scalar<53122>,
    minor_correction: Scalar<42507>,
    major_correction: Scalar<53283>,
    pieces_correction: Scalar<63172>,
    correction_gradient_gamma: Scalar<2197>,
    correction_gradient_delta: Scalar<2816>,
    pawns_correction_bonus: Scalar<3448>,
    minor_correction_bonus: Scalar<3643>,
    major_correction_bonus: Scalar<3725>,
    pieces_correction_bonus: Scalar<3487>,
    quiet_history_bonus_gamma: Scalar<9242>,
    quiet_history_bonus_delta: Scalar<1394>,
    noisy_history_bonus_gamma: Scalar<6489>,
    noisy_history_bonus_delta: Scalar<385>,
    quiet_continuation_bonus_gamma: Scalar<10262>,
    quiet_continuation_bonus_delta: Scalar<1772>,
    noisy_continuation_bonus_gamma: Scalar<9174>,
    noisy_continuation_bonus_delta: Scalar<1553>,
    quiet_history_penalty_gamma: Scalar<-10964>,
    quiet_history_penalty_delta: Scalar<-3622>,
    noisy_history_penalty_gamma: Scalar<-8954>,
    noisy_history_penalty_delta: Scalar<-2331>,
    quiet_continuation_penalty_gamma: Scalar<-10082>,
    quiet_continuation_penalty_delta: Scalar<-1355>,
    noisy_continuation_penalty_gamma: Scalar<-9464>,
    noisy_continuation_penalty_delta: Scalar<-1900>,
    improving_2: Scalar<3601>,
    improving_4: Scalar<4429>,
    fail_high_reduction_gamma: Scalar<469832>,
    fail_high_reduction_delta: Scalar<234248>,
    fail_low_reduction_gamma: Scalar<1473904>,
    fail_low_reduction_delta: Scalar<481719>,
    single_extension_margin_gamma: Scalar<2893>,
    single_extension_margin_delta: Scalar<2059>,
    double_extension_margin_gamma: Scalar<4826>,
    double_extension_margin_delta: Scalar<1173>,
    triple_extension_margin_gamma: Scalar<2600>,
    triple_extension_margin_delta: Scalar<636415>,
    null_move_pruning_gamma: Scalar<47917>,
    null_move_pruning_delta: Scalar<29724>,
    razoring_margin_theta: Scalar<39340>,
    razoring_margin_gamma: Scalar<97390>,
    razoring_margin_delta: Scalar<154303>,
    reverse_futility_margin_theta: Scalar<7786>,
    reverse_futility_margin_gamma: Scalar<41220>,
    reverse_futility_margin_delta: Scalar<36629>,
    reverse_futility_margin_improving: Scalar<-21363>,
    reverse_futility_margin_noisy_pv: Scalar<-4902>,
    reverse_futility_margin_cut: Scalar<-11169>,
    futility_margin_theta: Scalar<1541>,
    futility_margin_gamma: Scalar<133928>,
    futility_margin_delta: Scalar<232438>,
    futility_margin_is_pv: Scalar<30176>,
    futility_margin_was_pv: Scalar<26686>,
    futility_margin_gain: Scalar<5710>,
    futility_margin_killer: Scalar<27863>,
    futility_margin_check: Scalar<27551>,
    futility_margin_history: Scalar<18111>,
    futility_margin_counter: Scalar<20278>,
    futility_margin_improving: Scalar<16346>,
    see_pruning_theta: Scalar<-2392>,
    see_pruning_gamma: Scalar<-249649>,
    see_pruning_delta: Scalar<3922>,
    see_pruning_killer: Scalar<-31273>,
    see_pruning_history: Scalar<-10874>,
    see_pruning_counter: Scalar<-21552>,
    late_move_reduction_theta: Scalar<918>,
    late_move_reduction_gamma: Scalar<398>,
    late_move_reduction_delta: Scalar<2919>,
    late_move_reduction_baseline: Scalar<1219>,
    late_move_reduction_root: Scalar<-2384>,
    late_move_reduction_is_pv: Scalar<-2906>,
    late_move_reduction_was_pv: Scalar<-901>,
    late_move_reduction_killer: Scalar<-4802>,
    late_move_reduction_check: Scalar<-3988>,
    late_move_reduction_history: Scalar<-4127>,
    late_move_reduction_counter: Scalar<-5773>,
    late_move_reduction_improving: Scalar<-1763>,
    late_move_reduction_noisy_pv: Scalar<3242>,
    late_move_reduction_cut: Scalar<5715>,
    late_move_pruning_theta: Scalar<1772>,
    late_move_pruning_gamma: Scalar<1160>,
    late_move_pruning_delta: Scalar<4131>,
    late_move_pruning_baseline: Scalar<3056>,
    late_move_pruning_root: Scalar<4422>,
    late_move_pruning_is_pv: Scalar<4235>,
    late_move_pruning_was_pv: Scalar<3281>,
    late_move_pruning_check: Scalar<3917>,
    late_move_pruning_improving: Scalar<4657>,
    killer_move_bonus: Scalar<225177>,
    history_rating: Scalar<437376>,
    counter_rating: Scalar<680320>,
    winning_rating_gamma: Scalar<7102>,
    winning_rating_delta: Scalar<61864>,
    aspiration_window_start: Scalar<22194>,
    aspiration_window_gamma: Scalar<5599>,
    aspiration_window_delta: Scalar<6676>,
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

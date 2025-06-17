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
    moves_left_start: Param<737578>,
    moves_left_end: Param<9536>,
    soft_time_fraction: Param<2363>,
    hard_time_fraction: Param<2989>,
    score_trend_inertia: Param<23947>,
    pv_focus_gamma: Param<8015>,
    pv_focus_delta: Param<9239>,
    score_trend_magnitude: Param<2241>,
    score_trend_pivot: Param<233982>,
    history_bonus_gamma: Param<6422>,
    history_bonus_delta: Param<888>,
    history_penalty_gamma: Param<7057>,
    history_penalty_delta: Param<945>,
    null_move_reduction_gamma: Param<62307>,
    null_move_reduction_delta: Param<45897>,
    fail_high_reduction_gamma: Param<472196>,
    fail_high_reduction_delta: Param<267111>,
    fail_low_reduction_gamma: Param<1445660>,
    fail_low_reduction_delta: Param<405743>,
    single_extension_margin_gamma: Param<3542>,
    single_extension_margin_delta: Param<501>,
    double_extension_margin_gamma: Param<3840>,
    double_extension_margin_delta: Param<1677>,
    triple_extension_margin_gamma: Param<3725>,
    triple_extension_margin_delta: Param<581678>,
    razoring_margin_gamma: Param<342547>,
    razoring_margin_delta: Param<156037>,
    reverse_futility_margin_gamma: Param<60256>,
    reverse_futility_margin_delta: Param<30646>,
    futility_margin_gamma: Param<205136>,
    futility_margin_delta: Param<295248>,
    futility_pruning_threshold_gamma: Param<37074>,
    see_pruning_threshold_gamma: Param<241880>,
    late_move_reduction_gamma: Param<1668>,
    late_move_reduction_delta: Param<1245>,
    late_move_reduction_pv: Param<4513>,
    late_move_reduction_cut: Param<4263>,
    late_move_reduction_improving: Param<2781>,
    late_move_reduction_history: Param<4853>,
    late_move_reduction_killer: Param<4385>,
    late_move_reduction_check: Param<3861>,
    late_move_reduction_tt_noisy: Param<3915>,
    late_move_pruning_gamma: Param<2095>,
    late_move_pruning_delta: Param<5449>,
    killer_move_bonus: Param<261265>,
    noisy_gain_rating_gamma: Param<6274>,
    noisy_gain_rating_delta: Param<65825>,
    aspiration_window_start: Param<18960>,
    aspiration_window_gamma: Param<5104>,
    aspiration_window_delta: Param<5897>,
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

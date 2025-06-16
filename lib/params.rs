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
    moves_left_start: Param<839096>,
    moves_left_end: Param<9719>,
    soft_time_fraction: Param<2504>,
    hard_time_fraction: Param<2537>,
    score_trend_inertia: Param<19900>,
    pv_focus_gamma: Param<8831>,
    pv_focus_delta: Param<10075>,
    score_trend_magnitude: Param<2156>,
    score_trend_pivot: Param<226142>,
    history_bonus_gamma: Param<5707>,
    history_bonus_delta: Param<838>,
    history_penalty_gamma: Param<5948>,
    history_penalty_delta: Param<1098>,
    null_move_reduction_gamma: Param<63722>,
    null_move_reduction_delta: Param<50741>,
    fail_high_reduction_gamma: Param<482250>,
    fail_high_reduction_delta: Param<262090>,
    fail_low_reduction_gamma: Param<1459530>,
    fail_low_reduction_delta: Param<348202>,
    single_extension_margin_gamma: Param<3497>,
    single_extension_margin_delta: Param<795>,
    double_extension_margin_gamma: Param<4018>,
    double_extension_margin_delta: Param<1799>,
    triple_extension_margin_gamma: Param<3914>,
    triple_extension_margin_delta: Param<565867>,
    razoring_margin_gamma: Param<329600>,
    razoring_margin_delta: Param<156071>,
    reverse_futility_margin_gamma: Param<61581>,
    reverse_futility_margin_delta: Param<33657>,
    futility_margin_gamma: Param<209666>,
    futility_margin_delta: Param<271725>,
    futility_pruning_threshold_gamma: Param<34922>,
    see_pruning_threshold_gamma: Param<229451>,
    late_move_reduction_gamma: Param<1899>,
    late_move_reduction_delta: Param<513>,
    late_move_reduction_pv: Param<4332>,
    late_move_reduction_cut: Param<3959>,
    late_move_reduction_improving: Param<3810>,
    late_move_reduction_history: Param<4492>,
    late_move_reduction_killer: Param<4225>,
    late_move_reduction_check: Param<4156>,
    late_move_reduction_tt_noisy: Param<3948>,
    late_move_pruning_gamma: Param<2164>,
    late_move_pruning_delta: Param<5099>,
    killer_move_bonus: Param<292717>,
    noisy_gain_rating_gamma: Param<6598>,
    noisy_gain_rating_delta: Param<64815>,
    aspiration_window_start: Param<19978>,
    aspiration_window_gamma: Param<5064>,
    aspiration_window_delta: Param<5645>,
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

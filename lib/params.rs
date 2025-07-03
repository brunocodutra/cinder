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
    score_trend_pivot: Scalar<135241>,
    pawns_correction: Scalar<49839>,
    minor_correction: Scalar<45339>,
    major_correction: Scalar<48980>,
    pieces_correction: Scalar<61072>,
    correction_gradient_gamma: Scalar<1932>,
    correction_gradient_delta: Scalar<2829>,
    pawns_correction_bonus: Scalar<128431>,
    minor_correction_bonus: Scalar<119736>,
    major_correction_bonus: Scalar<120857>,
    pieces_correction_bonus: Scalar<124393>,
    history_bonus_quiet_gamma: Scalar<320738>,
    history_bonus_quiet_delta: Scalar<45168>,
    history_bonus_noisy_gamma: Scalar<211339>,
    history_bonus_noisy_delta: Scalar<10332>,
    history_penalty_quiet_gamma: Scalar<-379900>,
    history_penalty_quiet_delta: Scalar<-110331>,
    history_penalty_noisy_gamma: Scalar<-251402>,
    history_penalty_noisy_delta: Scalar<-90270>,
    continuation_bonus_quiet_gamma: Scalar<11220>,
    continuation_bonus_quiet_delta: Scalar<1877>,
    continuation_bonus_noisy_gamma: Scalar<8724>,
    continuation_bonus_noisy_delta: Scalar<1453>,
    continuation_penalty_quiet_gamma: Scalar<-9530>,
    continuation_penalty_quiet_delta: Scalar<-1366>,
    continuation_penalty_noisy_gamma: Scalar<-8736>,
    continuation_penalty_noisy_delta: Scalar<-2016>,
    fail_high_reduction_gamma: Scalar<437482>,
    fail_high_reduction_delta: Scalar<243415>,
    fail_low_reduction_gamma: Scalar<1469245>,
    fail_low_reduction_delta: Scalar<449990>,
    single_extension_margin_gamma: Scalar<2238>,
    single_extension_margin_delta: Scalar<2131>,
    double_extension_margin_gamma: Scalar<5071>,
    double_extension_margin_delta: Scalar<1239>,
    triple_extension_margin_gamma: Scalar<2485>,
    triple_extension_margin_delta: Scalar<600655>,
    null_move_pruning_gamma: Scalar<50188>,
    null_move_pruning_delta: Scalar<23384>,
    razoring_margin_scalar: Vector2<{ [135346, 103110] }>,
    razoring_margin_depth: Vector1<{ [45913] }>,
    reverse_futility_margin_scalar: Vector5<{ [40060, 42754, -28079, -4537, -11256] }>,
    reverse_futility_margin_depth: Vector4<{ [8031, 0, 0, 0] }>,
    reverse_futility_margin_improving: Vector3<{ [0, 0, 0] }>,
    reverse_futility_margin_noisy_pv: Vector2<{ [0, 0] }>,
    reverse_futility_margin_cut: Vector1<{ [0] }>,
    futility_margin_scalar: Vector7<{ [212895, 154546, 30719, 25270, 26365, 28997, 28016] }>,
    futility_margin_depth: Vector6<{ [1683, 0, 0, 0, 0, 0] }>,
    futility_margin_is_pv: Vector5<{ [0, 0, 0, 0, 0] }>,
    futility_margin_was_pv: Vector4<{ [0, 0, 0, 0] }>,
    futility_margin_improving: Vector3<{ [0, 0, 0] }>,
    futility_margin_is_killer: Vector2<{ [0, 0] }>,
    futility_margin_in_check: Vector1<{ [0] }>,
    futility_margin_gain: Scalar<4843>,
    futility_margin_history: Scalar<18508>,
    futility_margin_continuation: Scalar<22578>,
    see_pruning_margin_scalar: Vector3<{ [3559, -249438, -33588] }>,
    see_pruning_margin_depth: Vector2<{ [-2589, 0] }>,
    see_pruning_margin_is_killer: Vector1<{ [0] }>,
    see_pruning_history: Scalar<-11354>,
    see_pruning_continuation: Scalar<-20205>,
    late_move_reduction_scalar: Vector11<{ [4049, 406, 406, -3032, -903, -2337, -4796, -3682, 3459, 5921, -2298] }>,
    late_move_reduction_depth: Vector10<{ [0, 852, 0, 0, 0, 0, 0, 0, 0, 0] }>,
    late_move_reduction_index: Vector9<{ [0, 0, 0, 0, 0, 0, 0, 0, 0] }>,
    late_move_reduction_is_pv: Vector8<{ [0, 0, 0, 0, 0, 0, 0, 3032] }>,
    late_move_reduction_was_pv: Vector7<{ [0, 0, 0, 0, 0, 0, 903] }>,
    late_move_reduction_improving: Vector6<{ [0, 0, 0, 0, 0, 0] }>,
    late_move_reduction_is_killer: Vector5<{ [0, 0, 0, 0, 0] }>,
    late_move_reduction_gives_check: Vector4<{ [0, 0, 0, 0] }>,
    late_move_reduction_noisy_pv: Vector3<{ [0, 0, 0] }>,
    late_move_reduction_cut: Vector2<{ [0, 0] }>,
    late_move_reduction_is_root: Vector1<{ [0] }>,
    late_move_reduction_history: Scalar<-4138>,
    late_move_reduction_continuation: Scalar<-6114>,
    late_move_pruning_scalar: Vector7<{ [4011, 1041, 0, 0, 0, 0, 0] }>,
    late_move_pruning_depth: Vector6<{ [2067, 3945, 2918, 6488, 3805, 3999] }>,
    late_move_pruning_is_pv: Vector5<{ [0, 0, 0, 0, 0] }>,
    late_move_pruning_was_pv: Vector4<{ [0, 0, 0, 0] }>,
    late_move_pruning_improving: Vector3<{ [0, 0, 0] }>,
    late_move_pruning_in_check: Vector2<{ [0, 0] }>,
    late_move_pruning_is_root: Vector1<{ [0] }>,
    killer_move_bonus: Scalar<244544>,
    history_rating: Scalar<452601>,
    continuation_rating: Scalar<505784>,
    winning_rating_gamma: Scalar<5753>,
    winning_rating_delta: Scalar<69058>,
    aspiration_window_start: Scalar<18939>,
    aspiration_window_gamma: Scalar<5949>,
    aspiration_window_delta: Scalar<7779>,
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

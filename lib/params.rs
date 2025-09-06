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
            pub const BASE: i64 = 4096;

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
    value_scale: Constant1<{ [524288] }>,
    moves_left_start: Constant1<{ [729101] }>,
    moves_left_end: Constant1<{ [8541] }>,
    soft_time_fraction: Constant1<{ [3055] }>,
    hard_time_fraction: Constant1<{ [3501] }>,
    score_trend_inertia: Constant1<{ [28907] }>,
    pv_focus_gamma: Constant1<{ [6752] }>,
    pv_focus_delta: Constant1<{ [7735] }>,
    score_trend_magnitude: Constant1<{ [3439] }>,
    score_trend_pivot: Variable1<{ [115026] }>,
    pawns_correction: Variable1<{ [54481] }>,
    minor_correction: Variable1<{ [39651] }>,
    major_correction: Variable1<{ [47126] }>,
    pieces_correction: Variable1<{ [65980] }>,
    correction_gradient_depth: Variable2<{ [0, 2001] }>,
    correction_gradient_scalar: Variable1<{ [2655] }>,
    pawns_correction_bonus: Variable1<{ [7693] }>,
    minor_correction_bonus: Variable1<{ [7099] }>,
    major_correction_bonus: Variable1<{ [7795] }>,
    pieces_correction_bonus: Variable1<{ [7576] }>,
    history_bonus_depth: Variable2<{ [0, 21197] }>,
    history_bonus_scalar: Variable1<{ [2910] }>,
    continuation_bonus_depth: Variable2<{ [0, 26484] }>,
    continuation_bonus_scalar: Variable1<{ [3472] }>,
    history_penalty_depth: Variable2<{ [0, -24824] }>,
    history_penalty_scalar: Variable1<{ [-6302] }>,
    continuation_penalty_depth: Variable2<{ [0, -23279] }>,
    continuation_penalty_scalar: Variable1<{ [-2983] }>,
    probcut_margin_depth: Variable2<{ [0, 54585] }>,
    probcut_margin_scalar: Variable1<{ [831668] }>,
    single_extension_margin_depth: Variable2<{ [0, 2965] }>,
    single_extension_margin_scalar: Variable1<{ [2190] }>,
    double_extension_margin_depth: Variable2<{ [0, 4527] }>,
    double_extension_margin_scalar: Variable1<{ [1060] }>,
    triple_extension_margin_depth: Variable2<{ [0, 2364] }>,
    triple_extension_margin_scalar: Variable1<{ [586023] }>,
    null_move_reduction_gamma: Variable1<{ [37727] }>,
    null_move_reduction_delta: Variable1<{ [23294] }>,
    null_move_pruning_depth: Variable2<{ [2155, 39100] }>,
    null_move_pruning_scalar: Variable1<{ [-21267] }>,
    fail_low_pruning_depth: Variable2<{ [91057, 1420567] }>,
    fail_low_pruning_scalar: Variable1<{ [-535707] }>,
    fail_high_pruning_depth: Variable2<{ [129568, 259797] }>,
    fail_high_pruning_scalar: Variable1<{ [-209828] }>,
    razoring_margin_depth: Variable2<{ [33492, 92911] }>,
    razoring_margin_scalar: Variable1<{ [142927] }>,
    reverse_futility_margin_depth: Variable2<{ [5318, 51661] }>,
    reverse_futility_margin_scalar: Variable1<{ [35531] }>,
    reverse_futility_margin_improving: Variable1<{ [-22964] }>,
    futility_margin_depth: Variable2<{ [7282, 107051] }>,
    futility_margin_scalar: Variable1<{ [208800] }>,
    futility_margin_is_pv: Variable1<{ [29798] }>,
    futility_margin_was_pv: Variable1<{ [24775] }>,
    futility_margin_is_check: Variable1<{ [31032] }>,
    futility_margin_is_killer: Variable1<{ [27855] }>,
    futility_margin_improving: Variable1<{ [17947] }>,
    futility_margin_gain: Variable1<{ [4907] }>,
    noisy_see_pruning_depth: Variable2<{ [-8806, -184304] }>,
    noisy_see_pruning_scalar: Variable1<{ [35296] }>,
    quiet_see_pruning_depth: Variable2<{ [-37237, -3560] }>,
    quiet_see_pruning_scalar: Variable1<{ [35798] }>,
    see_pruning_is_killer: Variable1<{ [-27634] }>,
    late_move_reduction_depth: Variable3<{ [0, 917, 381] }>,
    late_move_reduction_index: Variable2<{ [0, 403] }>,
    late_move_reduction_scalar: Variable1<{ [3101] }>,
    late_move_reduction_baseline: Variable1<{ [1319] }>,
    late_move_reduction_is_root: Variable1<{ [-1028] }>,
    late_move_reduction_is_pv: Variable1<{ [-3115] }>,
    late_move_reduction_was_pv: Variable1<{ [-1082] }>,
    late_move_reduction_gives_check: Variable1<{ [-3625] }>,
    late_move_reduction_is_noisy_pv: Variable1<{ [2953] }>,
    late_move_reduction_is_killer: Variable1<{ [-5161] }>,
    late_move_reduction_cut: Variable1<{ [5292] }>,
    late_move_reduction_improving: Variable1<{ [-1477] }>,
    late_move_reduction_history: Variable1<{ [-3476] }>,
    late_move_reduction_counter: Variable1<{ [-5119] }>,
    late_move_pruning_depth: Variable2<{ [1839, 1304] }>,
    late_move_pruning_scalar: Variable1<{ [3695] }>,
    late_move_pruning_baseline: Variable1<{ [2982] }>,
    late_move_pruning_is_root: Variable1<{ [6690] }>,
    late_move_pruning_is_pv: Variable1<{ [4747] }>,
    late_move_pruning_was_pv: Variable1<{ [2974] }>,
    late_move_pruning_is_check: Variable1<{ [3764] }>,
    late_move_pruning_improving: Variable1<{ [3804] }>,
    improving_2: Variable2<{ [3532, 0] }>,
    improving_4: Variable1<{ [3875] }>,
    killer_move_bonus: Variable1<{ [211716] }>,
    history_rating: Variable1<{ [429813] }>,
    counter_rating: Variable1<{ [549452] }>,
    winning_rating_depth: Variable2<{ [0, 7031] }>,
    winning_rating_scalar: Variable1<{ [66332] }>,
    aspiration_window_baseline: Variable1<{ [22928] }>,
    aspiration_window_exponent: Variable2<{ [0, 5942] }>,
    aspiration_window_scalar: Variable1<{ [6179] }>,
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

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
    moves_left_start: Constant1<{ [722518] }>,
    moves_left_end: Constant1<{ [8696] }>,
    soft_time_fraction: Constant1<{ [3097] }>,
    hard_time_fraction: Constant1<{ [3499] }>,
    score_trend_inertia: Constant1<{ [28422] }>,
    pv_focus_gamma: Constant1<{ [6871] }>,
    pv_focus_delta: Constant1<{ [7706] }>,
    score_trend_magnitude: Constant1<{ [3602] }>,
    score_trend_pivot: Constant1<{ [118593] }>,
    pawn_values: Variable8<{ [170118, 214211, 217367, 237286, 235022, 238328, 225554, 205577] }>,
    knight_values: Variable8<{ [938545, 755638, 787837, 778845, 787083, 753004, 747059, 691997] }>,
    bishop_values: Variable8<{ [992960, 880573, 863465, 917657, 890968, 868464, 852696, 783403] }>,
    rook_values: Variable8<{ [1042912, 1279105, 1347928, 1367398, 1378899, 1295857, 1297156, 1158015] }>,
    queen_values: Variable8<{ [2300625, 2399430, 2603939, 2690447, 2482394, 2585993, 2571291, 2636932] }>,
    pawns_correction: Variable8<{ [57307, 56664, 57084, 55497, 57380, 56545, 57858, 57331] }>,
    minor_correction: Variable8<{ [40573, 40390, 39975, 40507, 40864, 40023, 40674, 40159] }>,
    major_correction: Variable8<{ [46121, 44722, 44921, 47733, 47147, 46742, 46859, 46781] }>,
    pieces_correction: Variable8<{ [66449, 66674, 67509, 65301, 65604, 69182, 66592, 66265] }>,
    pawns_correction_grad_depth: Variable2<{ [0, 3716] }>,
    pawns_correction_grad_scalar: Variable1<{ [4868] }>,
    minor_correction_grad_depth: Variable2<{ [0, 3689] }>,
    minor_correction_grad_scalar: Variable1<{ [4811] }>,
    major_correction_grad_depth: Variable2<{ [0, 3819] }>,
    major_correction_grad_scalar: Variable1<{ [4946] }>,
    pieces_correction_grad_depth: Variable2<{ [0, 3703] }>,
    pieces_correction_grad_scalar: Variable1<{ [4937] }>,
    history_bonus_depth: Variable2<{ [0, 21013] }>,
    history_bonus_scalar: Variable1<{ [2981] }>,
    continuation_bonus_depth: Variable2<{ [0, 27778] }>,
    continuation_bonus_scalar: Variable1<{ [3610] }>,
    history_penalty_depth: Variable2<{ [0, -24113] }>,
    history_penalty_scalar: Variable1<{ [-6364] }>,
    continuation_penalty_depth: Variable2<{ [0, -22556] }>,
    continuation_penalty_scalar: Variable1<{ [-3003] }>,
    probcut_margin_depth: Variable2<{ [0, 55486] }>,
    probcut_margin_scalar: Variable1<{ [863096] }>,
    single_extension_margin_depth: Variable2<{ [0, 3005] }>,
    single_extension_margin_scalar: Variable1<{ [2231] }>,
    double_extension_margin_depth: Variable2<{ [0, 4402] }>,
    double_extension_margin_scalar: Variable1<{ [1077] }>,
    triple_extension_margin_depth: Variable2<{ [0, 2465] }>,
    triple_extension_margin_scalar: Variable1<{ [578775] }>,
    null_move_reduction_gamma: Variable1<{ [37312] }>,
    null_move_reduction_delta: Variable1<{ [22474] }>,
    null_move_pruning_depth: Variable2<{ [2246, 41874] }>,
    null_move_pruning_scalar: Variable1<{ [-19940] }>,
    fail_low_pruning_depth: Variable2<{ [97338, 1404636] }>,
    fail_low_pruning_scalar: Variable1<{ [-537141] }>,
    fail_high_pruning_depth: Variable2<{ [133291, 264179] }>,
    fail_high_pruning_scalar: Variable1<{ [-210162] }>,
    razoring_margin_depth: Variable2<{ [34068, 90287] }>,
    razoring_margin_scalar: Variable1<{ [138577] }>,
    reverse_futility_margin_depth: Variable2<{ [5159, 51720] }>,
    reverse_futility_margin_scalar: Variable1<{ [38027] }>,
    reverse_futility_margin_improving: Variable1<{ [-23609] }>,
    futility_margin_depth: Variable2<{ [7042, 106329] }>,
    futility_margin_scalar: Variable1<{ [211834] }>,
    futility_margin_is_pv: Variable1<{ [29936] }>,
    futility_margin_was_pv: Variable1<{ [24107] }>,
    futility_margin_is_check: Variable1<{ [31404] }>,
    futility_margin_is_killer: Variable1<{ [27795] }>,
    futility_margin_improving: Variable1<{ [17340] }>,
    futility_margin_gain: Variable1<{ [4779] }>,
    noisy_see_pruning_depth: Variable2<{ [-9019, -190439] }>,
    noisy_see_pruning_scalar: Variable1<{ [37012] }>,
    quiet_see_pruning_depth: Variable2<{ [-37167, -3547] }>,
    quiet_see_pruning_scalar: Variable1<{ [34272] }>,
    see_pruning_is_killer: Variable1<{ [-28753] }>,
    late_move_reduction_depth: Variable3<{ [0, 922, 400] }>,
    late_move_reduction_index: Variable2<{ [0, 403] }>,
    late_move_reduction_scalar: Variable1<{ [3174] }>,
    late_move_reduction_baseline: Variable1<{ [1321] }>,
    late_move_reduction_is_root: Variable1<{ [-1049] }>,
    late_move_reduction_is_pv: Variable1<{ [-3031] }>,
    late_move_reduction_was_pv: Variable1<{ [-1068] }>,
    late_move_reduction_gives_check: Variable1<{ [-3731] }>,
    late_move_reduction_is_noisy_pv: Variable1<{ [2963] }>,
    late_move_reduction_is_killer: Variable1<{ [-4956] }>,
    late_move_reduction_cut: Variable1<{ [5180] }>,
    late_move_reduction_improving: Variable1<{ [-1491] }>,
    late_move_reduction_history: Variable1<{ [-3601] }>,
    late_move_reduction_counter: Variable1<{ [-4875] }>,
    late_move_pruning_depth: Variable2<{ [1785, 1320] }>,
    late_move_pruning_scalar: Variable1<{ [3749] }>,
    late_move_pruning_baseline: Variable1<{ [2800] }>,
    late_move_pruning_is_root: Variable1<{ [6447] }>,
    late_move_pruning_is_pv: Variable1<{ [4656] }>,
    late_move_pruning_was_pv: Variable1<{ [2819] }>,
    late_move_pruning_is_check: Variable1<{ [3825] }>,
    late_move_pruning_improving: Variable1<{ [3968] }>,
    improving_2: Variable2<{ [3600, 0] }>,
    improving_4: Variable1<{ [3719] }>,
    killer_move_bonus: Variable1<{ [217660] }>,
    history_rating: Variable1<{ [428411] }>,
    counter_rating: Variable1<{ [540851] }>,
    winning_rating_margin: Variable1<{ [-83032] }>,
    winning_rating_gain: Variable2<{ [0, 6860] }>,
    winning_rating_scalar: Variable1<{ [66584] }>,
    aspiration_window_baseline: Variable1<{ [23409] }>,
    aspiration_window_exponent: Variable2<{ [0, 5828] }>,
    aspiration_window_scalar: Variable1<{ [6339] }>,
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

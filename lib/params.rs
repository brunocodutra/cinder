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
    moves_left_start: Constant1<{ [704116] }>,
    moves_left_end: Constant1<{ [8404] }>,
    soft_time_fraction: Constant1<{ [3064] }>,
    hard_time_fraction: Constant1<{ [3489] }>,
    score_trend_inertia: Constant1<{ [28454] }>,
    pv_focus_gamma: Constant1<{ [6768] }>,
    pv_focus_delta: Constant1<{ [7711] }>,
    score_trend_magnitude: Constant1<{ [3420] }>,
    score_trend_pivot: Variable1<{ [114222] }>,
    pawn_values: Variable8<{ [166784, 207104, 212928, 234112, 243392, 231008, 217504, 204064] }>,
    knight_values: Variable8<{ [942688, 768160, 779456, 783104, 777984, 763008, 728384, 685440] }>,
    bishop_values: Variable8<{ [1013664, 860928, 876608, 886336, 878272, 852352, 831392, 787456] }>,
    rook_values: Variable8<{ [1017504, 1271392, 1321280, 1376672, 1369888, 1325472, 1303392, 1244864] }>,
    queen_values: Variable8<{ [2251488, 2503552, 2579648, 2662752, 2613600, 2541536, 2457248, 2592608] }>,
    pawns_correction: Variable1<{ [55273] }>,
    minor_correction: Variable1<{ [39438] }>,
    major_correction: Variable1<{ [47572] }>,
    pieces_correction: Variable1<{ [66471] }>,
    correction_gradient_depth: Variable2<{ [0, 2015] }>,
    correction_gradient_scalar: Variable1<{ [2639] }>,
    pawns_correction_bonus: Variable1<{ [7738] }>,
    minor_correction_bonus: Variable1<{ [7277] }>,
    major_correction_bonus: Variable1<{ [7869] }>,
    pieces_correction_bonus: Variable1<{ [7640] }>,
    history_bonus_depth: Variable2<{ [0, 20752] }>,
    history_bonus_scalar: Variable1<{ [2949] }>,
    continuation_bonus_depth: Variable2<{ [0, 26410] }>,
    continuation_bonus_scalar: Variable1<{ [3508] }>,
    history_penalty_depth: Variable2<{ [0, -24785] }>,
    history_penalty_scalar: Variable1<{ [-6399] }>,
    continuation_penalty_depth: Variable2<{ [0, -23147] }>,
    continuation_penalty_scalar: Variable1<{ [-2957] }>,
    probcut_margin_depth: Variable2<{ [0, 54886] }>,
    probcut_margin_scalar: Variable1<{ [843678] }>,
    single_extension_margin_depth: Variable2<{ [0, 2933] }>,
    single_extension_margin_scalar: Variable1<{ [2200] }>,
    double_extension_margin_depth: Variable2<{ [0, 4540] }>,
    double_extension_margin_scalar: Variable1<{ [1056] }>,
    triple_extension_margin_depth: Variable2<{ [0, 2363] }>,
    triple_extension_margin_scalar: Variable1<{ [591867] }>,
    null_move_reduction_gamma: Variable1<{ [37564] }>,
    null_move_reduction_delta: Variable1<{ [23309] }>,
    null_move_pruning_depth: Variable2<{ [2172, 39431] }>,
    null_move_pruning_scalar: Variable1<{ [-21133] }>,
    fail_low_pruning_depth: Variable2<{ [91401, 1389169] }>,
    fail_low_pruning_scalar: Variable1<{ [-532357] }>,
    fail_high_pruning_depth: Variable2<{ [129196, 263967] }>,
    fail_high_pruning_scalar: Variable1<{ [-210164] }>,
    razoring_margin_depth: Variable2<{ [33218, 94115] }>,
    razoring_margin_scalar: Variable1<{ [144310] }>,
    reverse_futility_margin_depth: Variable2<{ [5195, 51942] }>,
    reverse_futility_margin_scalar: Variable1<{ [35831] }>,
    reverse_futility_margin_improving: Variable1<{ [-23066] }>,
    futility_margin_depth: Variable2<{ [7220, 104348] }>,
    futility_margin_scalar: Variable1<{ [209451] }>,
    futility_margin_is_pv: Variable1<{ [30005] }>,
    futility_margin_was_pv: Variable1<{ [24510] }>,
    futility_margin_is_check: Variable1<{ [31054] }>,
    futility_margin_is_killer: Variable1<{ [27623] }>,
    futility_margin_improving: Variable1<{ [17552] }>,
    futility_margin_gain: Variable1<{ [4813] }>,
    noisy_see_pruning_depth: Variable2<{ [-8943, -182708] }>,
    noisy_see_pruning_scalar: Variable1<{ [35789] }>,
    quiet_see_pruning_depth: Variable2<{ [-37785, -3566] }>,
    quiet_see_pruning_scalar: Variable1<{ [35951] }>,
    see_pruning_is_killer: Variable1<{ [-28142] }>,
    late_move_reduction_depth: Variable3<{ [0, 946, 371] }>,
    late_move_reduction_index: Variable2<{ [0, 407] }>,
    late_move_reduction_scalar: Variable1<{ [3111] }>,
    late_move_reduction_baseline: Variable1<{ [1332] }>,
    late_move_reduction_is_root: Variable1<{ [-1045] }>,
    late_move_reduction_is_pv: Variable1<{ [-3092] }>,
    late_move_reduction_was_pv: Variable1<{ [-1082] }>,
    late_move_reduction_gives_check: Variable1<{ [-3643] }>,
    late_move_reduction_is_noisy_pv: Variable1<{ [2883] }>,
    late_move_reduction_is_killer: Variable1<{ [-5076] }>,
    late_move_reduction_cut: Variable1<{ [5394] }>,
    late_move_reduction_improving: Variable1<{ [-1500] }>,
    late_move_reduction_history: Variable1<{ [-3535] }>,
    late_move_reduction_counter: Variable1<{ [-5048] }>,
    late_move_pruning_depth: Variable2<{ [1785, 1291] }>,
    late_move_pruning_scalar: Variable1<{ [3727] }>,
    late_move_pruning_baseline: Variable1<{ [2882] }>,
    late_move_pruning_is_root: Variable1<{ [6612] }>,
    late_move_pruning_is_pv: Variable1<{ [4788] }>,
    late_move_pruning_was_pv: Variable1<{ [2975] }>,
    late_move_pruning_is_check: Variable1<{ [3691] }>,
    late_move_pruning_improving: Variable1<{ [3820] }>,
    improving_2: Variable2<{ [3629, 0] }>,
    improving_4: Variable1<{ [3802] }>,
    killer_move_bonus: Variable1<{ [213473] }>,
    history_rating: Variable1<{ [433765] }>,
    counter_rating: Variable1<{ [539996] }>,
    winning_rating_margin: Variable1<{ [-82471] }>,
    winning_rating_gain: Variable2<{ [0, 7024] }>,
    winning_rating_scalar: Variable1<{ [66044] }>,
    aspiration_window_baseline: Variable1<{ [22979] }>,
    aspiration_window_exponent: Variable2<{ [0, 5899] }>,
    aspiration_window_scalar: Variable1<{ [6337] }>,
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

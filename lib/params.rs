use crate::util::{ByteBuffer, TypedByteBuffer};
use std::{cell::SyncUnsafeCell, ops::Deref, slice::SliceIndex};

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

#[cfg(test)]
#[cfg(feature = "spsa")]
use proptest::{collection::vec, prelude::*};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "spsa", serde(into = "Vec<f32>", from = "Vec<f32>"))]
struct Param<const BYTES: ByteBuffer<32>> {
    #[cfg(feature = "spsa")]
    #[cfg_attr(test, strategy(vec(-1e3f32..=1e3f32, 8).prop_map(|vs| TypedByteBuffer::new(&vs))))]
    values: TypedByteBuffer<f32, 32>,
}

impl<const BYTES: ByteBuffer<32>> Param<BYTES> {
    const VALUES: TypedByteBuffer<f32, 32> = unsafe { TypedByteBuffer::from_bytes(BYTES) };

    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "spsa")]
            values: Self::VALUES,
        }
    }

    #[cfg(feature = "spsa")]
    pub fn perturb<I: IntoIterator<Item = f32>>(&self, perturbations: I) -> (Self, Self) {
        let (mut left, mut right) = (self.clone(), self.clone());
        let mut perturbations = perturbations.into_iter();
        for (i, c) in Self::VALUES.iter().enumerate() {
            let delta = c.abs() * perturbations.next().unwrap();
            left.values[i] += delta;
            right.values[i] -= delta;
        }

        (left, right)
    }

    #[cfg(feature = "spsa")]
    pub fn update<I: IntoIterator<Item = f32>>(&mut self, corrections: I) {
        let mut corrections = corrections.into_iter();
        for (i, c) in Self::VALUES.iter().enumerate() {
            let delta = c.abs() * corrections.next().unwrap();
            self.values[i] += delta;
        }
    }
}

impl<const BYTES: ByteBuffer<32>> const Default for Param<BYTES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const BYTES: ByteBuffer<32>> const Deref for Param<BYTES> {
    type Target = [f32];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        #[cfg(feature = "spsa")]
        {
            &self.values
        }

        #[cfg(not(feature = "spsa"))]
        {
            &Self::VALUES
        }
    }
}

#[cfg(feature = "spsa")]
impl<const BYTES: ByteBuffer<32>> From<Param<BYTES>> for Vec<f32> {
    fn from(param: Param<BYTES>) -> Self {
        param.to_vec()
    }
}

#[cfg(feature = "spsa")]
impl<const BYTES: ByteBuffer<32>> From<Vec<f32>> for Param<BYTES> {
    fn from(values: Vec<f32>) -> Self {
        Self {
            values: TypedByteBuffer::new(&values),
        }
    }
}

static PARAMS: SyncUnsafeCell<Params> = SyncUnsafeCell::new(Params::new());

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
    ($head:expr,) => { $head.len() };
    ($head:expr, $($tail:expr,)*) => {
        $head.len() + len!($($tail,)*)
    }
}

macro_rules! params {
    ($($name: ident: $value: expr,)*) => {
        #[derive(Debug, Default, Clone, PartialEq)]
        #[cfg_attr(test, derive(test_strategy::Arbitrary))]
        #[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
        #[cfg_attr(feature = "spsa", serde(deny_unknown_fields))]
        pub struct Params {
            $(
                #[cfg_attr(feature = "spsa", serde(default))]
                $name: Param<{ TypedByteBuffer::<f32, 32>::new(&$value).into_bytes() }>,
            )*
        }

        impl Params {
            const fn new() -> Self {
                Params {
                    $($name: Param::new(),)*
                }
            }
        }

        $(impl Params {
            /// This parameter's current value.
            #[inline(always)]
            pub const fn $name<R: [const] SliceIndex<[f32]>>(idx: R) -> &'static R::Output {
                unsafe { PARAMS.get().as_ref_unchecked().$name.get_unchecked(idx) }
            }
        })*

        #[cfg(feature = "spsa")]
        impl Params {
            /// Total number of parameters.
            pub const LEN: usize = len!($($value,)*);

            /// Initializes the global params.
            pub const fn init(self) {
                unsafe { *PARAMS.get().as_mut_unchecked() = self }
            }

            /// Perturb parameters in both positive and negative directions.
            ///
            /// # Panic
            ///
            /// Panics if the `perturbations`'s length is less than [`Self::LEN`].
            pub fn perturb<I: IntoIterator<Item = f32>>(&self, perturbations: I) -> (Self, Self) {
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
            pub fn update<I: IntoIterator<Item = f32>>(&mut self, corrections: I) {
                let mut corrections = corrections.into_iter();
                $(self.$name.update(&mut corrections);)*
            }
        }
    };
}

params! {
    moves_left_start: [181.82031],
    moves_left_end: [2.1155741],
    soft_time_fraction: [0.74782526],
    hard_time_fraction: [0.8520899],
    score_trend_inertia: [0.12889749],
    pv_focus_gamma: [-1.7051414],
    pv_focus_delta: [1.8834248],
    score_trend_magnitude: [0.88112456],
    score_trend_pivot: [27.66006],
    improving: [0.0, 0.877628, 0.8956914, 1.7754363],
    pawn_values: [40.519474, 52.25906, 51.715137, 57.109646, 58.426537, 57.42907, 56.835117, 50.27611],
    knight_values: [235.48207, 183.96709, 196.70047, 186.74135, 196.87659, 184.97841, 178.65738, 167.56061],
    bishop_values: [248.33437, 210.2506, 214.3851, 222.43594, 217.54062, 211.84268, 210.45186, 193.63986],
    rook_values: [251.46643, 319.81677, 330.07047, 332.5627, 326.2085, 317.42154, 320.32602, 291.919],
    queen_values: [550.06976, 591.42017, 639.1168, 660.70746, 602.29407, 642.625, 618.7207, 650.29224],
    pawns_correction: [13.9612255, 13.542381, 13.816187, 13.852693, 13.642615, 13.643646, 14.470145, 13.788407],
    minor_correction: [10.053518, 9.934302, 9.878027, 9.762766, 9.911766, 10.118631, 9.663956, 9.620695],
    major_correction: [11.514736, 10.546854, 11.066393, 11.091913, 11.323167, 11.085468, 10.905323, 11.576518],
    pieces_correction: [16.310757, 16.175463, 16.022593, 16.411938, 16.629448, 16.610157, 16.325285, 16.480825],
    pawns_correction_bonus: [1.0172647],
    minor_correction_bonus: [0.9866924],
    major_correction_bonus: [1.0692323],
    pieces_correction_bonus: [1.0458503],
    history_bonus_depth: [0.0, 5.10541],
    history_bonus_scalar: [0.7390443],
    continuation_bonus_depth: [0.0, 6.7304516],
    continuation_bonus_scalar: [0.8707493],
    history_penalty_depth: [0.0, -5.963275],
    history_penalty_scalar: [-1.5365195],
    continuation_penalty_depth: [0.0, -5.410611],
    continuation_penalty_scalar: [-0.71752197],
    probcut_margin_depth: [0.0, 13.636732],
    probcut_margin_scalar: [202.28221],
    single_extension_margin_depth: [0.0, 0.7197645],
    single_extension_margin_scalar: [0.5525035],
    double_extension_margin_depth: [0.0, 1.0644989],
    double_extension_margin_scalar: [0.26553062],
    triple_extension_margin_depth: [0.0, 0.59066916],
    triple_extension_margin_scalar: [142.8579],
    tt_cut_halfmove_limit: [89.12544],
    tb_cut_depth_bonus: [4.003424],
    flp_margin_depth: [23.603306, 333.9373],
    flp_margin_scalar: [-130.91628],
    fhp_margin_depth: [32.579494, 64.37981],
    fhp_margin_scalar: [-50.41566],
    nmp_margin_depth: [0.5432876, 9.914622],
    nmp_margin_scalar: [-4.9237638],
    nmr_gamma: [0.12892021],
    nmr_delta: [0.118562244],
    nmr_limit: [3.056344],
    nmr_fraction: [0.2591698],
    razoring_depth: [8.350244, 22.024878],
    razoring_scalar: [32.191067],
    rfp_margin_depth: [1.231198, 12.166299],
    rfp_margin_scalar: [9.188515],
    rfp_margin_improving: [-5.809523],
    fut_margin_depth: [1.7374123, 25.445139],
    fut_margin_scalar: [52.476643],
    fut_margin_is_pv: [7.0300865],
    fut_margin_was_pv: [5.921309],
    fut_margin_is_check: [7.6741014],
    fut_margin_is_killer: [6.756299],
    fut_margin_improving: [4.2783628],
    fut_margin_gain: [1.1965386],
    nsp_margin_depth: [-2.1696308, -46.909004],
    nsp_margin_scalar: [8.471801],
    qsp_margin_depth: [-9.09216, -0.86537164],
    qsp_margin_scalar: [8.387581],
    sp_margin_is_killer: [-7.096274],
    lmp_depth: [0.43936145, 0.32097837],
    lmp_scalar: [0.9306138],
    lmp_baseline: [0.7050455],
    lmp_is_root: [1.6330844],
    lmp_is_pv: [1.1405051],
    lmp_was_pv: [0.7051773],
    lmp_is_check: [0.944297],
    lmp_improving: [0.95657784],
    lmr_depth: [0.0, 0.23226961, 0.09825783],
    lmr_index: [0.0, 0.10209553],
    lmr_scalar: [0.76791525],
    lmr_baseline: [0.31703383],
    lmr_is_root: [-0.26581413],
    lmr_is_pv: [-0.75085205],
    lmr_was_pv: [-0.27456057],
    lmr_gives_check: [-0.9073583],
    lmr_is_noisy_pv: [0.7247483],
    lmr_is_killer: [-1.2642287],
    lmr_cut: [1.2749865],
    lmr_improving: [-0.37780857],
    lmr_history: [-0.8556145],
    lmr_counter: [-1.2006621],
    killer_rating: [52.804733],
    history_rating: [106.61823],
    counter_rating: [124.09749],
    winning_rating_margin: [-20.491241],
    winning_rating_gain: [0.0, 1.6685247],
    winning_rating_scalar: [15.832105],
    aw_baseline: [2048., 512., 128., 32., 8., 4.9224224, 4.9224224, 4.9224224],
    aw_gamma: [1.4124817],
    aw_delta: [1.2470169],
    aw_fail_low_blend: [0.51431525],
    aw_fail_high_reduction: [1.0188276],
}

#[cfg(test)]
#[cfg(feature = "spsa")]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn perturbing_updates_params(
        p: Params,
        #[strategy(vec(-1f32..=1f32, Params::LEN))] d: Vec<f32>,
    ) {
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

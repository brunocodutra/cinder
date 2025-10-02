use crate::{nnue::Layer, util::Aligned};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::{Debug, Deref, DerefMut};

/// The feature transformer accumulator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable, Deref, DerefMut)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Accumulator")]
#[repr(transparent)]
pub struct Accumulator(
    #[cfg_attr(test, map(|vs: [i8; Self::LEN]| Aligned(vs.map(i16::from))))]
    Aligned<[i16; Accumulator::LEN]>,
);

impl Default for Accumulator {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl Layer for Accumulator {
    const LEN: usize = 2048;
    type Neuron = i16;
}

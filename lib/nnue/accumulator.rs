use crate::{nnue::Layer, simd::Aligned};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::{Debug, Deref, DerefMut};

/// The feature transformer accumulator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable, Deref, DerefMut)]
#[debug("Accumulator")]
pub struct Accumulator(Aligned<[i16; Accumulator::LEN]>);

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

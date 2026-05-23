use crate::{nnue::Layer, simd::Aligned};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;
use std::ops::{Deref, DerefMut};

/// The feature transformer accumulator.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Zeroable)]
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

impl Deref for Accumulator {
    type Target = Aligned<[i16; Accumulator::LEN]>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Accumulator {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

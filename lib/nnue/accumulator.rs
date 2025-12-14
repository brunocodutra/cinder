use crate::{nnue::Layer, simd::Aligned};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;
use std::ops::{Deref, DerefMut};

/// The feature transformer accumulator.
#[derive(Debug, Clone, Hash, Zeroable)]
#[derive_const(Eq, PartialEq)]
#[debug("Accumulator")]
pub struct Accumulator(Aligned<[i16; Accumulator::LEN]>);

impl const Default for Accumulator {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl Layer for Accumulator {
    const LEN: usize = 2048;
    type Neuron = i16;
}

impl const Deref for Accumulator {
    type Target = Aligned<[i16; Accumulator::LEN]>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl const DerefMut for Accumulator {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

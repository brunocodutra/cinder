use crate::util::AlignTo64;
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::{Debug, Deref, DerefMut};

/// The feature transformer accumulator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable, Deref, DerefMut)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Accumulator")]
#[repr(transparent)]
pub struct Accumulator(
    #[cfg_attr(test, map(|vs: [i8; Self::LEN]| AlignTo64(vs.map(i16::from))))]
    AlignTo64<[i16; Accumulator::LEN]>,
);

impl Accumulator {
    pub const LEN: usize = 2048;
}

impl Default for Accumulator {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

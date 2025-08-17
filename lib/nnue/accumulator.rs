use crate::{chess::Phase, util::AlignTo64};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::{Debug, Deref, DerefMut};

/// The material accumulator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable, Deref, DerefMut)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Material")]
#[repr(transparent)]
pub struct Material(
    #[cfg_attr(test, map(|vs: [i8; Self::LEN]| vs.map(i32::from)))] [i32; Material::LEN],
);

impl Material {
    pub const LEN: usize = Phase::LEN;
}

impl Default for Material {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

/// The positional accumulator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable, Deref, DerefMut)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Positional")]
#[repr(transparent)]
pub struct Positional(
    #[cfg_attr(test, map(|vs: [i8; Self::LEN]| AlignTo64(vs.map(i16::from))))]
    AlignTo64<[i16; Positional::LEN]>,
);

impl Positional {
    pub const LEN: usize = 768;
}

impl Default for Positional {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

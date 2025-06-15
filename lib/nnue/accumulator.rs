use crate::chess::Phase;
use crate::util::{AlignTo64, Integer};
use derive_more::with_trait::{Debug, Deref, DerefMut};

/// The material accumulator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Deref, DerefMut)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Material")]
#[repr(transparent)]
pub struct Material(
    #[cfg_attr(test, map(|vs: [i8; Self::LEN]| vs.map(i32::from)))] [i32; Self::LEN],
);

impl Material {
    pub const LEN: usize = Phase::MAX as usize + 1;
}

impl Default for Material {
    #[inline(always)]
    fn default() -> Self {
        Material([0; Self::LEN])
    }
}

/// The positional accumulator.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Deref, DerefMut)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Positional")]
#[repr(transparent)]
pub struct Positional(
    #[cfg_attr(test, map(|vs: [i8; Self::LEN]| AlignTo64(vs.map(i16::from))))]
    AlignTo64<[i16; Self::LEN]>,
);

impl Positional {
    pub const LEN: usize = 768;
}

impl Default for Positional {
    #[inline(always)]
    fn default() -> Self {
        Positional(AlignTo64([0; Self::LEN]))
    }
}

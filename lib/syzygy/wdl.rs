use crate::search::{Ply, Score};
use crate::util::{Binary, Bits, Integer};
use crate::{nnue::Value, syzygy::Dtz};
use std::ops::Neg;

/// The possible outcomes of a final [`Position`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(i8)]
pub enum Wdl {
    /// Unconditional loss.
    Loss = -2,
    /// Loss that can be saved by the 50-move rule.
    BlessedLoss = -1,
    /// Unconditional draw.
    Draw = 0,
    /// Win that can be frustrated by the 50-move rule.
    CursedWin = 1,
    /// Unconditional win.
    Win = 2,
}

unsafe impl Integer for Wdl {
    type Repr = i8;
    const MIN: Self::Repr = Self::Loss as _;
    const MAX: Self::Repr = Self::Win as _;
}

impl Wdl {
    /// Convert to [`Score`].
    #[inline(always)]
    pub fn to_score(self, ply: Ply) -> Score {
        match self {
            Wdl::Win => Score::new(Value::MAX) - ply,
            Wdl::Loss => Score::new(Value::MIN) + ply,
            _ => Score::new(0),
        }
    }
}

impl Neg for Wdl {
    type Output = Wdl;

    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.get())
    }
}

/// Converts [`Dtz`] to a [`Wdl`].
///
/// | DTZ  | WDL          |
/// | ---- | ------------ |
/// | -1   | Loss         |
/// | -101 | Blessed loss |
/// | 0    | Draw         |
/// | 101  | Cursed win   |
/// | 1    | Win          |
impl From<Dtz> for Wdl {
    #[inline(always)]
    fn from(dtz: Dtz) -> Self {
        match dtz.get() {
            ..-100 => Wdl::BlessedLoss,
            -100..0 => Wdl::Loss,
            0 => Wdl::Draw,
            1..101 => Wdl::Win,
            101.. => Wdl::CursedWin,
        }
    }
}

impl Binary for Wdl {
    type Bits = Bits<u8, 3>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        Bits::new((self.get() - Self::MIN + 1).cast())
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        Wdl::new(bits.cast::<i8>() + Self::MIN - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn wdl_has_an_equivalent_dtz(wdl: Wdl) {
        assert_eq!(Wdl::from(Dtz::from(wdl)), wdl);
    }

    #[proptest]
    fn decoding_encoded_wdl_is_an_identity(wdl: Wdl) {
        assert_eq!(Wdl::decode(wdl.encode()), wdl);
    }

    #[proptest]
    fn decoding_encoded_optional_wdl_is_an_identity(wdl: Option<Wdl>) {
        assert_eq!(Option::decode(wdl.encode()), wdl);
    }
}

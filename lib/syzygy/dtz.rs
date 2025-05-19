use crate::syzygy::Wdl;
use crate::util::{Binary, Bits, Bounded, Integer};
use derive_more::with_trait::{Constructor, Neg};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Constructor, Neg)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct DtzRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Self as Integer>::Repr);

unsafe impl Integer for DtzRepr {
    type Repr = i16;
    const MIN: Self::Repr = -Self::MAX;
    const MAX: Self::Repr = 1023;
}

/// DTZ<sub>50</sub>. Based on the distance to zeroing of the half-move clock.
///
/// Zeroing the half-move clock while keeping the game-theoretical result in
/// hand guarantees making progress, so min-maxing `Dtz` values guarantees
/// achieving the optimal outcome under the 50-move rule.
///
/// | DTZ               | WDL          | |
/// | ----------------- | ------------ | - |
/// | `-100 <= n <= -1` | Loss         | Unconditional loss (assuming the 50-move counter is zero). Zeroing move can be forced in `-n` plies. |
/// | `n < -100`        | Blessed loss | Loss, but draw under the 50-move rule. A zeroing move can be forced in `-n` plies or `-n - 100` plies (if a later phase is responsible for the blessing). |
/// | 0                 | Draw         | |
/// | `100 < n`         | Cursed win   | Win, but draw under the 50-move rule. A zeroing move can be forced in `n` or `n - 100` plies (if a later phase is responsible for the curse). |
/// | `1 <= n <= 100`   | Win          | Unconditional win (assuming the 50-move counter is zero). Zeroing move can be forced in `n` plies. |
pub type Dtz = Bounded<DtzRepr>;

impl Dtz {
    /// Increases the absolute non-zero value by `plies`.
    #[inline(always)]
    pub fn stretch(self, plies: u16) -> Dtz {
        self + self.signum() as i32 * plies as i32
    }
}

/// Converts [`Wdl`] to a [`Dtz`].
///
/// | WDL          | DTZ  |
/// | ------------ | ---- |
/// | Loss         | -1   |
/// | Blessed loss | -101 |
/// | Draw         | 0    |
/// | Cursed win   | 101  |
/// | Win          | 1    |
impl From<Wdl> for Dtz {
    #[inline(always)]
    fn from(wdl: Wdl) -> Self {
        match wdl {
            Wdl::Loss => Dtz::new(-1),
            Wdl::BlessedLoss => Dtz::new(-101),
            Wdl::Draw => Dtz::new(0),
            Wdl::CursedWin => Dtz::new(101),
            Wdl::Win => Dtz::new(1),
        }
    }
}

impl Binary for Dtz {
    type Bits = Bits<u16, 11>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        Bits::new((self.get() - Self::MIN + 1).cast())
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        Dtz::new(bits.cast::<i16>() + Self::MIN - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn stretching_dtz_increases_magnitude(dtz: Dtz, p: u16) {
        assert_eq!(dtz.stretch(p).signum(), dtz.signum());
    }

    #[proptest]
    fn decoding_encoded_dtz_is_an_identity(dtz: Dtz) {
        assert_eq!(Dtz::decode(dtz.encode()), dtz);
    }

    #[proptest]
    fn decoding_encoded_optional_dtz_is_an_identity(dtz: Option<Dtz>) {
        assert_eq!(Option::decode(dtz.encode()), dtz);
    }
}

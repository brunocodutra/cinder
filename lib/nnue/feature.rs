use crate::chess::{Color, File, Perspective, Piece, Side, Square};
use crate::util::Int;

/// The king's bucket.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq, Ord, PartialOrd)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Bucket(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Bucket as Int>::Repr);

impl Bucket {
    pub const LEN: usize = Self::MAX as usize + 1;
}

unsafe impl const Int for Bucket {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 31;
}

/// A bucketed feature set with horizontal mirroring.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Feature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Feature as Int>::Repr);

unsafe impl const Int for Feature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

impl Feature {
    /// The total number of different features.
    pub const LEN: usize = 768 * Bucket::LEN / 2;

    /// Constructs a [`Feature`].
    #[inline(always)]
    pub const fn new(side: Color, ksq: Square, piece: Piece, sq: Square) -> Self {
        let chirality = Side::from(ksq.file() < File::E);
        let bucket = Self::bucket(side, ksq.perspective(chirality)).cast::<u16>();
        let psq = 64 * piece.perspective(side).cast::<u16>()
            + sq.perspective(side).perspective(chirality).cast::<u16>();

        Int::new(psq + 768 * bucket)
    }

    /// Constructs a [`Feature`].
    #[inline(always)]
    pub const fn bucket(side: Color, ksq: Square) -> Bucket {
        #[rustfmt::skip]
        const BUCKETS: [u8; 64] = [
            16, 17, 18, 19,  3,  2,  1,  0,
            20, 21, 22, 23,  7,  6,  5,  4,
            24, 25, 26, 27, 11, 10,  9,  8,
            24, 25, 26, 27, 11, 10,  9,  8,
            28, 29, 30, 31, 15, 14, 13, 12,
            28, 29, 30, 31, 15, 14, 13, 12,
            28, 29, 30, 31, 15, 14, 13, 12,
            28, 29, 30, 31, 15, 14, 13, 12,
        ];

        Int::new(BUCKETS[ksq.perspective(side).cast::<usize>()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{Flip, Mirror};
    use test_strategy::proptest;

    #[test]
    fn len_counts_total_number_of_features() {
        assert_eq!(Feature::LEN, Feature::iter().len());
    }

    #[proptest]
    fn is_unique_to_perspective(c: Color, ksq: Square, p: Piece, sq: Square) {
        assert_ne!(Feature::new(c, ksq, p, sq), Feature::new(!c, ksq, p, sq));
    }

    #[proptest]
    fn is_vertically_symmetric(c: Color, ksq: Square, p: Piece, sq: Square) {
        assert_eq!(
            Feature::new(c, ksq, p, sq),
            Feature::new(c.flip(), ksq.flip(), p.flip(), sq.flip())
        );
    }

    #[proptest]
    fn is_horizontally_symmetric(c: Color, ksq: Square, p: Piece, sq: Square) {
        assert_eq!(
            Feature::new(c, ksq, p, sq),
            Feature::new(c, ksq.mirror(), p, sq.mirror())
        );
    }
}

use crate::chess::{Color, File, Perspective, Piece, Side, Square};
use crate::util::Integer;

/// The king's bucket.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Bucket(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Self as Integer>::Repr);

unsafe impl Integer for Bucket {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 15;
}

/// A bucketed feature set with horizontal mirroring.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Feature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Self as Integer>::Repr);

unsafe impl Integer for Feature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

impl Feature {
    /// The total number of different features.
    pub const LEN: usize = 8 * 768;

    /// Constructs a [`Feature`].
    #[inline(always)]
    pub fn new(side: Color, ksq: Square, piece: Piece, sq: Square) -> Self {
        let chirality = Side::from(ksq.file() < File::E);
        let bucket = Self::bucket(side, ksq.perspective(chirality)).cast::<u16>();
        let psq = 64 * piece.perspective(side).cast::<u16>()
            + sq.perspective(side).perspective(chirality).cast::<u16>();

        Integer::new(psq + 768 * bucket)
    }

    /// Constructs a [`Feature`].
    #[inline(always)]
    pub fn bucket(side: Color, ksq: Square) -> Bucket {
        #[rustfmt::skip]
        const BUCKETS: [u8; 64] = [
             8,  8,  9,  9, 1, 1, 0, 0,
            10, 10, 11, 11, 3, 3, 2, 2,
            12, 12, 13, 13, 5, 5, 4, 4,
            12, 12, 13, 13, 5, 5, 4, 4,
            14, 14, 15, 15, 7, 7, 6, 6,
            14, 14, 15, 15, 7, 7, 6, 6,
            14, 14, 15, 15, 7, 7, 6, 6,
            14, 14, 15, 15, 7, 7, 6, 6,
        ];

        Integer::new(BUCKETS[ksq.perspective(side).cast::<usize>()])
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

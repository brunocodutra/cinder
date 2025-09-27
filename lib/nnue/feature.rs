use crate::chess::{Color, File, Perspective, Piece, Side, Square};
use crate::util::Integer;
use bytemuck::Zeroable;

/// The king's bucket.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Bucket(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Bucket as Integer>::Repr);

impl Bucket {
    pub const LEN: usize = Self::MAX as usize + 1;
}

unsafe impl Integer for Bucket {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 23;
}

/// A bucketed feature set with horizontal mirroring.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Feature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Feature as Integer>::Repr);

unsafe impl Integer for Feature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

impl Feature {
    /// The total number of different features.
    pub const LEN: usize = 768 * Bucket::LEN / 2;

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
            12, 13, 14, 15,  3,  2,  1,  0,
            16, 17, 18, 19,  7,  6,  5,  4,
            20, 20, 21, 21,  9,  9,  8,  8,
            20, 20, 21, 21,  9,  9,  8,  8,
            22, 22, 23, 23, 11, 11, 10, 10,
            22, 22, 23, 23, 11, 11, 10, 10,
            22, 22, 23, 23, 11, 11, 10, 10,
            22, 22, 23, 23, 11, 11, 10, 10,
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

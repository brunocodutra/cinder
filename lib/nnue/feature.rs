use crate::chess::{Color, File, Perspective, Piece, Side, Square};
use crate::util::{Assume, Int, Num};
use std::ops::{Index, IndexMut};

/// The king's bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Bucket(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Bucket as Num>::Repr);

impl Bucket {
    pub const LEN: usize = Self::MAX as usize + 1;
}

const unsafe impl Num for Bucket {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 31;
}

const unsafe impl Int for Bucket {}

const impl<T> Index<Bucket> for [T; Bucket::MAX as usize + 1] {
    type Output = T;

    #[inline(always)]
    fn index(&self, b: Bucket) -> &Self::Output {
        self.get(b.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Bucket> for [T; Bucket::MAX as usize + 1] {
    #[inline(always)]
    fn index_mut(&mut self, b: Bucket) -> &mut Self::Output {
        self.get_mut(b.cast::<usize>()).assume()
    }
}

/// A bucketed feature set with horizontal mirroring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Feature(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Feature as Num>::Repr);

const unsafe impl Num for Feature {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = Self::LEN as Self::Repr - 1;
}

const unsafe impl Int for Feature {}

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

        Num::new(psq + 768 * bucket)
    }

    /// Constructs a [`Feature`].
    #[inline(always)]
    pub fn bucket(side: Color, ksq: Square) -> Bucket {
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

        Num::new(BUCKETS[ksq.perspective(side)])
    }
}

const impl<T> Index<Feature> for [T; Feature::MAX as usize + 1] {
    type Output = T;

    #[inline(always)]
    fn index(&self, f: Feature) -> &Self::Output {
        self.get(f.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Feature> for [T; Feature::MAX as usize + 1] {
    #[inline(always)]
    fn index_mut(&mut self, f: Feature) -> &mut Self::Output {
        self.get_mut(f.cast::<usize>()).assume()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{Flip, Mirror};
    use test_strategy::proptest;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn len_counts_total_number_of_features() {
        assert_eq!(Feature::LEN, Feature::iter().len());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn is_unique_to_perspective(c: Color, ksq: Square, p: Piece, sq: Square) {
        assert_ne!(Feature::new(c, ksq, p, sq), Feature::new(!c, ksq, p, sq));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn is_vertically_symmetric(c: Color, ksq: Square, p: Piece, sq: Square) {
        assert_eq!(
            Feature::new(c, ksq, p, sq),
            Feature::new(c.flip(), ksq.flip(), p.flip(), sq.flip())
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn is_horizontally_symmetric(c: Color, ksq: Square, p: Piece, sq: Square) {
        assert_eq!(
            Feature::new(c, ksq, p, sq),
            Feature::new(c, ksq.mirror(), p, sq.mirror())
        );
    }
}

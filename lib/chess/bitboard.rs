use crate::chess::{Butterfly, File, Flip, Rank, Square};
use crate::simd::*;
use crate::util::{Assume, Int, Num};
use bytemuck::{Pod, Zeroable, zeroed};
use derive_more::with_trait::{Constructor, Debug, From};
use std::fmt::{self, Formatter, Write};
use std::{iter::FusedIterator, ops::*};

/// A set of squares on a chess board.
#[derive(Copy, Hash, Zeroable, Pod, Constructor, From)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Bitboard(<Bitboard as Num>::Repr);

const unsafe impl Num for Bitboard {
    type Repr = u64;
    const MIN: Self::Repr = u64::MIN;
    const MAX: Self::Repr = u64::MAX;
}

const unsafe impl Int for Bitboard {}

const impl Bitboard {
    /// An empty board.
    #[inline(always)]
    pub fn empty() -> Self {
        zeroed()
    }

    /// A full board.
    #[inline(always)]
    pub fn full() -> Self {
        Bitboard::new(0xFFFFFFFFFFFFFFFF)
    }

    /// Border squares.
    #[inline(always)]
    pub fn border() -> Self {
        Bitboard::new(0xFF818181818181FF)
    }

    /// Light squares.
    #[inline(always)]
    pub fn light() -> Self {
        Bitboard::new(0x55AA55AA55AA55AA)
    }

    /// Dark squares.
    #[inline(always)]
    pub fn dark() -> Self {
        Bitboard::new(0xAA55AA55AA55AA55)
    }

    /// Fills out squares on a bitboard.
    ///
    /// Starting from a square, fills out the squares by stepping on the board in each direction.
    /// Movement in a direction stops when an occupied square is reached.
    ///
    /// # Example
    /// ```
    /// # use cinder::chess::*;
    /// assert_eq!(
    ///     Vec::from_iter(Bitboard::fill(Square::E2, &[(-1, 2), (1, -1)], Square::C6.bitboard())),
    ///     vec![Square::F1, Square::E2, Square::D4, Square::C6]
    /// );
    /// ```
    #[inline(always)]
    pub fn fill(sq: Square, steps: &[(i8, i8)], occupied: Bitboard) -> Self {
        let mut bitboard = sq.bitboard();

        let mut i = steps.len();
        while i > 0 {
            i -= 1;
            let (df, dr) = steps[i];
            let mut sq = sq;
            while let Some((file, rank)) = Option::zip(
                (sq.file().get() + df).convert(),
                (sq.rank().get() + dr).convert(),
            ) {
                sq = Square::new(file, rank);
                bitboard = bitboard.with(sq);
                if occupied.contains(sq) {
                    break;
                }
            }
        }

        bitboard
    }

    /// Bitboard with squares in line with two other squares.
    ///
    /// # Example
    /// ```
    /// # use cinder::chess::*;
    /// assert_eq!(
    ///     Vec::from_iter(Bitboard::line(Square::B4, Square::E1)),
    ///     vec![Square::E1, Square::D2, Square::C3, Square::B4, Square::A5]
    /// );
    /// ```
    #[inline(always)]
    pub fn line(whence: Square, whither: Square) -> Self {
        static LINES: Butterfly<Bitboard> = const {
            let mut lines: Butterfly<Bitboard> = zeroed();

            for wc in Square::iter() {
                for wt in Square::iter() {
                    let df = wt.file() - wc.file();
                    let dr = wt.rank() - wc.rank();
                    if df == 0 && dr == 0 {
                        lines[wc][wt] = wc.bitboard();
                    } else if df == 0 {
                        lines[wc][wt] = wc.file().bitboard();
                    } else if dr == 0 {
                        lines[wc][wt] = wc.rank().bitboard();
                    } else if df.abs() == dr.abs() {
                        let steps = [(df.signum(), dr.signum()), (-df.signum(), -dr.signum())];
                        let bb = Bitboard::fill(wc, &steps, Bitboard::empty());
                        lines[wc][wt] = bb;
                    }
                }
            }

            lines
        };

        LINES[whence][whither]
    }

    /// Bitboard with squares in the open segment between two squares.
    ///
    /// # Example
    /// ```
    /// # use cinder::chess::*;
    /// assert_eq!(
    ///     Vec::from_iter(Bitboard::segment(Square::B4, Square::E1)),
    ///     vec![Square::D2, Square::C3]
    /// );
    /// ```
    #[inline(always)]
    pub fn segment(whence: Square, whither: Square) -> Self {
        static SEGMENTS: Butterfly<Bitboard> = const {
            let mut segments: Butterfly<Bitboard> = zeroed();

            for wc in Square::iter() {
                for wt in Square::iter() {
                    let df = wt.file() - wc.file();
                    let dr = wt.rank() - wc.rank();
                    if df == 0 || dr == 0 || df.abs() == dr.abs() {
                        let steps = [(df.signum(), dr.signum())];
                        let bb = Bitboard::fill(wc, &steps, wt.bitboard());
                        segments[wc][wt] = bb.without(wc).without(wt);
                    }
                }
            }

            segments
        };

        SEGMENTS[whence][whither]
    }

    /// The number of [`Square`]s in the set.
    #[inline(always)]
    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Whether the board is empty.
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self == Self::empty()
    }

    /// Whether this [`Square`] is in the set.
    #[inline(always)]
    pub fn contains(self, sq: Square) -> bool {
        self & sq.bitboard() != zeroed()
    }

    /// Adds a [`Square`] to this bitboard.
    #[inline(always)]
    pub fn with(self, sq: Square) -> Self {
        sq.bitboard().bitor(self)
    }

    /// Removes a [`Square`]s from this bitboard.
    #[inline(always)]
    pub fn without(self, sq: Square) -> Self {
        sq.bitboard().not().bitand(self)
    }

    /// Rotates the bitboard.
    #[inline(always)]
    pub fn rotate_left(self, n: u32) -> Self {
        Bitboard(self.0.rotate_left(n))
    }

    /// Rotates the bitboard.
    #[inline(always)]
    pub fn rotate_right(self, n: u32) -> Self {
        Bitboard(self.0.rotate_right(n))
    }

    /// An iterator over the [`Square`]s in this bitboard.
    #[inline(always)]
    pub fn iter(self) -> Squares {
        Squares::new(self)
    }
}

impl Debug for Bitboard {
    #[coverage(off)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char('\n')?;
        for rank in Rank::iter().rev() {
            for file in File::iter() {
                let sq = Square::new(file, rank);
                f.write_char(if self.contains(sq) { '■' } else { '◻' })?;
                f.write_char(if file < File::H { ' ' } else { '\n' })?;
            }
        }

        Ok(())
    }
}

const impl Deref for Bitboard {
    type Target = u64;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const impl Not for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(self.0.not())
    }
}

const impl BitAnd for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.bitand(rhs.0)
    }
}

const impl BitAnd<u64> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: u64) -> Self::Output {
        Self(self.0.bitand(rhs))
    }
}

impl<T: MaskElement> BitAnd<M<T, 64>> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: M<T, 64>) -> Self::Output {
        Self(self.0.bitand(rhs.to_bitmask()))
    }
}

impl<T: MaskElement> BitAnd<Mask<T, 64>> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Mask<T, 64>) -> Self::Output {
        Self(self.0.bitand(rhs.to_bitmask()))
    }
}

const impl BitAndAssign for Bitboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.bitand_assign(rhs.0);
    }
}

const impl BitAndAssign<u64> for Bitboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: u64) {
        self.0.bitand_assign(rhs);
    }
}

impl<T: MaskElement> BitAndAssign<M<T, 64>> for Bitboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: M<T, 64>) {
        self.0.bitand_assign(rhs.to_bitmask());
    }
}

impl<T: MaskElement> BitAndAssign<Mask<T, 64>> for Bitboard {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Mask<T, 64>) {
        self.0.bitand_assign(rhs.to_bitmask());
    }
}

const impl BitOr for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.bitor(rhs.0)
    }
}

const impl BitOr<u64> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: u64) -> Self::Output {
        Self(self.0.bitor(rhs))
    }
}

impl<T: MaskElement> BitOr<M<T, 64>> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: M<T, 64>) -> Self::Output {
        Self(self.0.bitor(rhs.to_bitmask()))
    }
}

impl<T: MaskElement> BitOr<Mask<T, 64>> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Mask<T, 64>) -> Self::Output {
        Self(self.0.bitor(rhs.to_bitmask()))
    }
}

const impl BitOrAssign for Bitboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.bitor_assign(rhs.0);
    }
}

const impl BitOrAssign<u64> for Bitboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: u64) {
        self.0.bitor_assign(rhs);
    }
}

impl<T: MaskElement> BitOrAssign<M<T, 64>> for Bitboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: M<T, 64>) {
        self.0.bitor_assign(rhs.to_bitmask());
    }
}

impl<T: MaskElement> BitOrAssign<Mask<T, 64>> for Bitboard {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Mask<T, 64>) {
        self.0.bitor_assign(rhs.to_bitmask());
    }
}

const impl BitXor for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.bitxor(rhs.0)
    }
}

const impl BitXor<u64> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: u64) -> Self::Output {
        Self(self.0.bitxor(rhs))
    }
}

impl<T: MaskElement> BitXor<M<T, 64>> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: M<T, 64>) -> Self::Output {
        Self(self.0.bitxor(rhs.to_bitmask()))
    }
}

impl<T: MaskElement> BitXor<Mask<T, 64>> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Mask<T, 64>) -> Self::Output {
        Self(self.0.bitxor(rhs.to_bitmask()))
    }
}

const impl BitXorAssign for Bitboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.bitxor_assign(rhs.0);
    }
}

const impl BitXorAssign<u64> for Bitboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: u64) {
        self.0.bitxor_assign(rhs);
    }
}

impl<T: MaskElement> BitXorAssign<M<T, 64>> for Bitboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: M<T, 64>) {
        self.0.bitxor_assign(rhs.to_bitmask());
    }
}

impl<T: MaskElement> BitXorAssign<Mask<T, 64>> for Bitboard {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Mask<T, 64>) {
        self.0.bitxor_assign(rhs.to_bitmask());
    }
}

const impl Shl<u32> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0.shl(rhs))
    }
}

const impl ShlAssign<u32> for Bitboard {
    #[inline(always)]
    fn shl_assign(&mut self, rhs: u32) {
        self.0.shl_assign(rhs);
    }
}

const impl Shr<u32> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0.shr(rhs))
    }
}

const impl ShrAssign<u32> for Bitboard {
    #[inline(always)]
    fn shr_assign(&mut self, rhs: u32) {
        self.0.shr_assign(rhs);
    }
}

const impl Flip for Bitboard {
    /// Flips all squares in the set.
    #[inline(always)]
    fn flip(self) -> Self {
        Self(self.0.swap_bytes())
    }
}

const impl From<File> for Bitboard {
    #[inline(always)]
    fn from(f: File) -> Self {
        f.bitboard()
    }
}

const impl From<Rank> for Bitboard {
    #[inline(always)]
    fn from(r: Rank) -> Self {
        r.bitboard()
    }
}

const impl From<Square> for Bitboard {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        sq.bitboard()
    }
}

impl<T: MaskElement> From<M<T, 64>> for Bitboard {
    #[inline(always)]
    fn from(mask: M<T, 64>) -> Self {
        Bitboard(mask.to_bitmask())
    }
}

impl<T: MaskElement> From<Mask<T, 64>> for Bitboard {
    #[inline(always)]
    fn from(mask: Mask<T, 64>) -> Self {
        Bitboard(mask.to_bitmask())
    }
}

#[cfg(target_feature = "avx512f")]
impl<T: MaskElement> From<Bitboard> for M<T, 64> {
    #[inline(always)]
    fn from(bb: Bitboard) -> Self {
        M::from_bitmask(bb.0)
    }
}

const impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = Squares;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        Squares::new(self)
    }
}

/// An iterator over the [`Square`]s in a [`Bitboard`].
#[derive(Debug, Constructor)]
pub struct Squares(Bitboard);

const impl Squares {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

const impl Iterator for Squares {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            None
        } else {
            let sq: Square = self.0.trailing_zeros().convert().assume();
            self.0 ^= sq.bitboard();
            Some(sq)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for Squares {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }
}

impl FusedIterator for Squares {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashSet, fmt::Debug};
    use test_strategy::proptest;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn empty_constructs_board_with_no_squares() {
        assert_eq!(Bitboard::empty().iter().count(), 0);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn full_constructs_board_with_all_squares() {
        assert_eq!(Bitboard::full().iter().count(), 64);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn border_constructs_bitboard_with_first_rank_eighth_rank_a_file_h_file() {
        assert_eq!(
            Bitboard::border(),
            Rank::First.bitboard()
                | Rank::Eighth.bitboard()
                | File::A.bitboard()
                | File::H.bitboard()
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn line_contains_both_squares(a: Square, b: Square) {
        assert_eq!(
            Bitboard::line(a, b).contains(a),
            Bitboard::line(a, b).contains(b)
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn line_degenerates_to_point(sq: Square) {
        assert_eq!(Bitboard::line(sq, sq), sq.bitboard());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn line_contains_segment(a: Square, b: Square) {
        assert_eq!(
            Bitboard::line(a, b) & Bitboard::segment(a, b),
            Bitboard::segment(a, b),
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn segment_does_not_contain_whence(a: Square, b: Square) {
        assert!(!Bitboard::segment(a, b).contains(a));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn segment_does_not_contain_whither(a: Square, b: Square) {
        assert!(!Bitboard::segment(a, b).contains(b));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn light_bitboards_contains_light_squares() {
        assert!(
            Bitboard::light()
                .iter()
                .all(|sq| (sq.file().get() + sq.rank().get()) % 2 != 0)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn dark_bitboards_contains_dark_squares() {
        assert!(
            Bitboard::dark()
                .iter()
                .all(|sq| (sq.file().get() + sq.rank().get()) % 2 == 0)
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn squares_are_either_light_or_dark() {
        assert_eq!(Bitboard::light() ^ Bitboard::dark(), Bitboard::full());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn len_returns_number_of_squares_on_the_board(bb: Bitboard) {
        assert_eq!(bb.len() as u32, bb.count_ones());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    #[expect(clippy::len_zero)]
    fn is_empty_returns_whether_there_are_squares_on_the_board(bb: Bitboard) {
        assert_eq!(bb.is_empty(), bb.len() == 0);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn contains_checks_whether_square_is_on_the_board(bb: Bitboard) {
        for sq in bb {
            assert!(bb.contains(sq));
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn with_adds_square_to_set(bb: Bitboard, sq: Square) {
        assert!(bb.with(sq).contains(sq));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn without_removes_square_to_set(bb: Bitboard, sq: Square) {
        assert!(!bb.without(sq).contains(sq));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn intersection_returns_squares_in_both_sets(a: Bitboard, b: Bitboard) {
        let c = a.bitand(b);
        for sq in Square::iter() {
            assert_eq!(c.contains(sq), a.contains(sq) && b.contains(sq));
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn union_returns_squares_in_either_set(a: Bitboard, b: Bitboard) {
        let c = a.bitor(b);
        for sq in Square::iter() {
            assert_eq!(c.contains(sq), a.contains(sq) || b.contains(sq));
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn flipping_a_bitboard_flips_every_square(bb: Bitboard) {
        assert_eq!(
            HashSet::<Square>::from_iter(bb.flip()),
            HashSet::<Square>::from_iter(bb.iter().map(Square::flip))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn can_iterate_over_squares_in_a_bitboard(bb: Bitboard, sq: Square) {
        let v = Vec::from_iter(bb);
        assert_eq!(bb.iter().len(), v.len());
        assert_eq!(bb.contains(sq), v.contains(&sq));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn bitboard_can_be_created_from_file(f: File) {
        assert_eq!(Bitboard::from(f), f.bitboard());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn bitboard_can_be_created_from_rank(r: Rank) {
        assert_eq!(Bitboard::from(r), r.bitboard());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn bitboard_can_be_created_from_square(sq: Square) {
        assert_eq!(Bitboard::from(sq), sq.bitboard());
    }
}

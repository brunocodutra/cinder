use super::{File, Rank};
use crate::util::{Binary, Bits};
use bitvec::{field::BitField, order::Lsb0, view::BitView};
use derive_more::{DebugCustom, Display, Error};
use proptest::sample::select;
use shakmaty as sm;
use std::convert::{Infallible, TryFrom, TryInto};
use std::num::TryFromIntError;
use test_strategy::Arbitrary;
use vampirc_uci::UciSquare;

/// Denotes a square on the chess board.
#[derive(DebugCustom, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Arbitrary)]
#[debug(fmt = "{}", self)]
#[display(fmt = "{}{}", "self.file()", "self.rank()")]
pub struct Square(#[strategy(select(sm::Square::ALL.as_ref()))] sm::Square);

impl Square {
    /// Constructs [`Square`] from a pair of [`File`] and [`Rank`].
    pub fn new(f: File, r: Rank) -> Self {
        Square(sm::Square::from_coords(f.into(), r.into()))
    }

    /// Constructs [`Square`] from index.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not in the range (0..64).
    pub fn from_index(i: u8) -> Self {
        i.try_into().unwrap()
    }

    /// This squares's index in the range (0..64).
    ///
    /// Squares are ordered from a1 = 0 to h8 = 63, files then ranks, so b1 = 2 and a2 = 8.
    pub fn index(&self) -> u8 {
        (*self).into()
    }

    /// Returns an iterator over [`Square`]s ordered by [index][`Square::index`].
    pub fn iter() -> impl DoubleEndedIterator<Item = Self> + ExactSizeIterator {
        sm::Square::ALL.into_iter().map(Square)
    }

    /// This square's [`File`].
    pub fn file(&self) -> File {
        self.0.file().into()
    }

    /// This square's [`Rank`].
    pub fn rank(&self) -> Rank {
        self.0.rank().into()
    }

    /// Mirrors this square's [`Rank`].
    pub fn mirror(&self) -> Self {
        self.0.flip_vertical().into()
    }
}

impl Binary for Square {
    type Register = Bits<u8, 6>;
    type Error = Infallible;

    fn encode(&self) -> Self::Register {
        self.index().view_bits::<Lsb0>().into()
    }

    fn decode(register: Self::Register) -> Result<Self, Self::Error> {
        Ok(Square::from_index(register.load()))
    }
}

/// The reason why converting [`Square`] from index failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display(fmt = "expected integer in the range `(0..64)`")]
pub struct SquareOutOfRange;

impl From<TryFromIntError> for SquareOutOfRange {
    fn from(_: TryFromIntError) -> Self {
        SquareOutOfRange
    }
}

impl TryFrom<u8> for Square {
    type Error = SquareOutOfRange;

    fn try_from(i: u8) -> Result<Self, Self::Error> {
        Ok(Square(i.try_into()?))
    }
}

impl From<Square> for u8 {
    fn from(s: Square) -> u8 {
        s.0.into()
    }
}

#[doc(hidden)]
impl From<Square> for UciSquare {
    fn from(s: Square) -> Self {
        UciSquare {
            file: s.file().into(),
            rank: s.rank().index() + 1,
        }
    }
}

#[doc(hidden)]
impl From<UciSquare> for Square {
    fn from(s: UciSquare) -> Self {
        Square::new(s.file.try_into().unwrap(), (s.rank - 1).try_into().unwrap())
    }
}

#[doc(hidden)]
impl From<sm::Square> for Square {
    fn from(s: sm::Square) -> Self {
        Square(s)
    }
}

#[doc(hidden)]
impl From<Square> for sm::Square {
    fn from(s: Square) -> Self {
        s.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use test_strategy::proptest;

    #[proptest]
    fn square_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Square>>(), size_of::<Square>());
    }

    #[proptest]
    fn new_constructs_square_from_pair_of_file_and_rank(s: Square) {
        assert_eq!(Square::new(s.file(), s.rank()), s);
    }

    #[proptest]
    fn iter_returns_iterator_over_files_in_order() {
        assert_eq!(
            Square::iter().collect::<Vec<_>>(),
            (0..=63).map(Square::from_index).collect::<Vec<_>>()
        );
    }

    #[proptest]
    fn iter_returns_double_ended_iterator() {
        assert_eq!(
            Square::iter().rev().collect::<Vec<_>>(),
            (0..=63).rev().map(Square::from_index).collect::<Vec<_>>()
        );
    }

    #[proptest]
    fn iter_returns_iterator_of_exact_size() {
        assert_eq!(Square::iter().len(), 64);
    }

    #[proptest]
    fn decoding_encoded_square_is_an_identity(s: Square) {
        assert_eq!(Square::decode(s.encode()), Ok(s));
    }

    #[proptest]
    fn decoding_square_never_fails(b: Bits<u8, 6>) {
        assert!(Square::decode(b).is_ok());
    }

    #[proptest]
    fn square_has_an_index(s: Square) {
        assert_eq!(s.index().try_into(), Ok(s));
    }

    #[proptest]
    fn square_has_a_mirror_on_the_same_file(s: Square) {
        assert_eq!(s.mirror(), Square::new(s.file(), s.rank().mirror()));
    }

    #[proptest]
    fn from_index_constructs_square_by_index(#[strategy(0u8..64)] i: u8) {
        assert_eq!(Square::from_index(i).index(), i);
    }

    #[proptest]
    #[should_panic]
    fn from_index_panics_if_index_out_of_range(#[strategy(64u8..)] i: u8) {
        Square::from_index(i);
    }

    #[proptest]
    fn converting_square_from_index_out_of_range_fails(#[strategy(64u8..)] i: u8) {
        assert_eq!(Square::try_from(i), Err(SquareOutOfRange));
    }

    #[proptest]
    fn square_is_ordered_by_index(a: Square, b: Square) {
        assert_eq!(a < b, a.index() < b.index());
    }

    #[proptest]
    fn square_has_an_equivalent_vampirc_uci_representation(s: Square) {
        assert_eq!(Square::from(<UciSquare as From<Square>>::from(s)), s);
    }

    #[proptest]
    fn square_has_an_equivalent_shakmaty_representation(s: Square) {
        assert_eq!(Square::from(sm::Square::from(s)), s);
    }
}
use crate::chess::{Bitboard, Perspective, Piece, Rank, Role, Square, Squares};
use crate::util::{Assume, Binary, Bits, Integer};
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::{num::NonZeroU16, ops::RangeBounds};

/// A chess move.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Move(NonZeroU16);

impl Move {
    #[inline(always)]
    fn bits<R: RangeBounds<u32>>(&self, r: R) -> Bits<u16, 16> {
        self.encode().slice(r)
    }

    /// Constructs a regular move.
    #[inline(always)]
    pub fn regular(whence: Square, whither: Square, promotion: Option<Role>) -> Self {
        let mut bits = Bits::<u16, 16>::default();
        bits.push(whence.encode());
        bits.push(whither.encode());

        match promotion {
            None => bits.push(Bits::<u8, 4>::new(0b0000)),
            Some(r) => {
                bits.push(Bits::<u8, 2>::new(0b10));
                bits.push(Bits::<u8, 2>::new(r.get() - 1));
            }
        }

        Move(bits.convert().assume())
    }

    /// Constructs a capture move.
    #[inline(always)]
    pub fn capture(whence: Square, whither: Square, promotion: Option<Role>) -> Self {
        let mut m = Self::regular(whence, whither, promotion);
        m.0 |= 0b100;
        m
    }

    /// The source [`Square`].
    #[inline(always)]
    pub fn whence(&self) -> Square {
        Square::decode(self.bits(10..).pop())
    }

    /// The destination [`Square`].
    #[inline(always)]
    pub fn whither(&self) -> Square {
        Square::decode(self.bits(4..).pop())
    }

    /// The promotion specifier.
    #[inline(always)]
    pub fn promotion(&self) -> Option<Role> {
        if self.is_promotion() {
            Some(Role::new(self.bits(..2).cast::<u8>() + 1))
        } else {
            None
        }
    }

    /// Whether this is a capture move.
    #[inline(always)]
    pub fn is_capture(&self) -> bool {
        self.bits(2..=2) != Bits::new(0)
    }

    /// Whether this is a promotion move.
    #[inline(always)]
    pub fn is_promotion(&self) -> bool {
        self.bits(3..=3) != Bits::new(0)
    }

    /// Whether this move is neither a capture nor a promotion.
    #[inline(always)]
    pub fn is_quiet(&self) -> bool {
        self.bits(2..=3) == Bits::new(0)
    }
}

impl Debug for Move {
    #[coverage(off)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self, f)?;

        if self.is_capture() {
            f.write_char('x')?;
        }

        Ok(())
    }
}

impl Display for Move {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.whence(), f)?;
        Display::fmt(&self.whither(), f)?;

        if let Some(r) = self.promotion() {
            Display::fmt(&r, f)?;
        }

        Ok(())
    }
}

impl Binary for Move {
    type Bits = Bits<u16, 16>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        self.0.convert().assume()
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        Move(bits.convert().assume())
    }
}

/// A collection of [`Move`]s originating from a given [`Square`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, filter(!#whither.contains(#base.whence())))]
pub struct MovePack {
    base: Move,
    whither: Bitboard,
}

impl MovePack {
    /// A pack of regular moves.
    #[inline(always)]
    pub fn regular(piece: Piece, whence: Square, whither: Bitboard) -> Self {
        use {Rank::*, Role::*};
        let base = if piece.role() == Pawn && whence.rank().perspective(piece.color()) == Seventh {
            Move::regular(whence, whence.flip(), Some(Knight))
        } else {
            Move::regular(whence, whence.flip(), None)
        };

        MovePack { base, whither }
    }

    /// A pack of capture moves.
    #[inline(always)]
    pub fn capture(piece: Piece, whence: Square, whither: Bitboard) -> Self {
        let mut moves = Self::regular(piece, whence, whither);
        moves.base.0 |= 0b100;
        moves
    }

    /// The source [`Square`].
    #[inline(always)]
    pub fn whence(&self) -> Square {
        self.base.whence()
    }

    /// The destination [`Square`]s.
    #[inline(always)]
    pub fn whither(&self) -> Bitboard {
        self.whither
    }

    /// Whether the moves in this set are captures.
    #[inline(always)]
    pub fn is_capture(&self) -> bool {
        self.base.is_capture()
    }

    /// Whether the moves in this set are promotions.
    #[inline(always)]
    pub fn is_promotion(&self) -> bool {
        self.base.is_promotion()
    }

    /// Whether the moves in this set are neither captures nor promotions.
    #[inline(always)]
    pub fn is_quiet(&self) -> bool {
        self.base.is_quiet()
    }

    /// An iterator over the [`Move`]s in this bitboard.
    #[inline(always)]
    pub fn iter(&self) -> MovePackIter {
        MovePackIter::new(*self)
    }
}

impl IntoIterator for MovePack {
    type Item = Move;
    type IntoIter = MovePackIter;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        MovePackIter::new(self)
    }
}

/// An iterator over the [`Move`]s in a [`MovePack`].
#[derive(Debug)]
pub struct MovePackIter {
    base: Move,
    whither: Squares,
}

impl MovePackIter {
    #[inline(always)]
    fn new(set: MovePack) -> Self {
        MovePackIter {
            base: set.base,
            whither: set.whither.iter(),
        }
    }
}

impl Iterator for MovePackIter {
    type Item = Move;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.base.is_promotion() {
            let mask = 0b1111111111111100;
            let promotion = [0b11, 0b00, 0b01, 0b10][(self.base.0.get() & !mask) as usize];
            self.base.0 = <NonZeroU16 as Integer>::new(self.base.0.get() & mask) | promotion;
        }

        if matches!(self.base.promotion(), None | Some(Role::Queen)) {
            let whither = self.whither.next()?;
            let bits = (self.base.0.get() & 0b1111110000001111) | ((whither as u16) << 4);
            self.base.0 = <NonZeroU16 as Integer>::new(bits);
        }

        Some(self.base)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl ExactSizeIterator for MovePackIter {
    #[inline(always)]
    fn len(&self) -> usize {
        match self.base.promotion() {
            None => self.whither.len(),
            Some(r) => 4 * self.whither.len() + (r.get() - Role::Knight.get()) as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::sample::select;
    use std::mem::size_of;
    use test_strategy::proptest;

    #[test]
    fn move_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Move>>(), size_of::<Move>());
    }

    #[proptest]
    fn decoding_encoded_move_is_an_identity(m: Move) {
        assert_eq!(Move::decode(m.encode()), m);
    }

    #[proptest]
    fn capture_move_can_be_constructed(
        wc: Square,
        #[filter(#wc != #wt)] wt: Square,
        #[strategy(select(&[Role::Knight, Role::Bishop, Role::Rook, Role::Queen]))] p: Role,
    ) {
        assert!(Move::capture(wc, wt, Some(p)).is_capture());
    }

    #[proptest]
    fn quiet_move_can_be_constructed(wc: Square, #[filter(#wc != #wt)] wt: Square) {
        assert!(Move::regular(wc, wt, None).is_quiet());
    }

    #[proptest]
    fn promotion_move_can_be_constructed(
        wc: Square,
        #[filter(#wc != #wt)] wt: Square,
        #[strategy(select(&[Role::Knight, Role::Bishop, Role::Rook, Role::Queen]))] p: Role,
    ) {
        assert!(Move::regular(wc, wt, Some(p)).is_promotion());
    }

    #[proptest]
    fn promotions_are_never_quiet(
        wc: Square,
        #[filter(#wc != #wt)] wt: Square,
        #[strategy(select(&[Role::Knight, Role::Bishop, Role::Rook, Role::Queen]))] p: Role,
    ) {
        assert!(!Move::regular(wc, wt, Some(p)).is_quiet());
    }

    #[proptest]
    fn captures_are_never_quiet(
        wc: Square,
        #[filter(#wc != #wt)] wt: Square,
        #[strategy(select(&[Role::Knight, Role::Bishop, Role::Rook, Role::Queen]))] p: Role,
    ) {
        assert!(!Move::capture(wc, wt, None).is_quiet());
        assert!(!Move::capture(wc, wt, Some(p)).is_quiet());
    }

    #[proptest]
    fn can_iterate_moves_in_set(ml: MovePack) {
        let v = Vec::from_iter(ml);
        assert_eq!(ml.iter().len(), v.len());
    }

    #[proptest]
    fn all_moves_in_set_are_of_the_same_type(ml: MovePack) {
        for m in ml {
            assert_eq!(m.is_promotion(), ml.is_promotion());
            assert_eq!(m.is_capture(), ml.is_capture());
            assert_eq!(m.is_quiet(), ml.is_quiet());
        }
    }

    #[proptest]
    fn all_moves_in_set_are_of_the_same_source_square(ml: MovePack) {
        for m in ml {
            assert_eq!(m.whence(), ml.whence());
        }
    }
}

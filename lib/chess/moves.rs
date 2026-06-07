use crate::{chess::*, simd::*, util::*};
use bytemuck::{ZeroableInOption, zeroed};
use derive_more::with_trait::{Deref, DerefMut, IntoIterator};
use std::fmt::{self, Debug, Display, Formatter, Write};
use std::{convert::Infallible, iter::FusedIterator, num::NonZeroU16};

#[cfg(test)]
use proptest::{collection::vec, prelude::*};

/// A chess move.
#[derive(Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, filter(#self.is_promotion() || #self.encode().slice(12..14) == zeroed()))]
#[repr(transparent)]
pub struct Move(NonZeroU16);

unsafe impl ZeroableInOption for Move {}

const impl Move {
    /// Constructs a regular move.
    #[inline(always)]
    pub fn regular(whence: Square, whither: Square, promotion: Option<Role>) -> Self {
        let mut bits = Bits::<u16, 16>::default();

        match promotion {
            None => bits.push(Bits::<u8, 4>::new(0b0000)),
            Some(r) => {
                bits.push(Bits::<u8, 2>::new(0b01));
                bits.push(Bits::<u8, 2>::new(r.get() - 1));
            }
        }

        bits.push(whither.encode());
        bits.push(whence.encode());
        Move(bits.convert().assume())
    }

    /// Constructs a capture move.
    #[inline(always)]
    pub fn capture(whence: Square, whither: Square, promotion: Option<Role>) -> Self {
        let mut m = Self::regular(whence, whither, promotion);
        m.0 |= 0b1000000000000000;
        m
    }

    /// The source [`Square`].
    #[inline(always)]
    pub fn whence(self) -> Square {
        Square::decode(self.encode().slice(..6).pop())
    }

    /// The destination [`Square`].
    #[inline(always)]
    pub fn whither(self) -> Square {
        Square::decode(self.encode().slice(6..).pop())
    }

    /// The promotion specifier.
    #[inline(always)]
    pub fn promotion(self) -> Option<Role> {
        if self.is_promotion() {
            Some(Role::new(self.encode().slice(12..14).cast::<u8>() + 1))
        } else {
            None
        }
    }

    /// Whether this is a capture move.
    #[inline(always)]
    pub fn is_capture(self) -> bool {
        self.encode().slice(15..=15) != zeroed()
    }

    /// Whether this is a promotion move.
    #[inline(always)]
    pub fn is_promotion(self) -> bool {
        self.encode().slice(14..=14) != zeroed()
    }

    /// Whether this move is neither a capture nor a promotion.
    #[inline(always)]
    pub fn is_quiet(self) -> bool {
        self.encode().slice(14..=15) == zeroed()
    }

    /// Whether this move is not quiet.
    #[inline(always)]
    pub fn is_noisy(self) -> bool {
        !self.is_quiet()
    }
}

impl Debug for Move {
    #[coverage(off)]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)?;

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

const impl Binary for Move {
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

/// A collection of [`Move`]s paired with their relative rating.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deref, DerefMut, IntoIterator)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Moves(
    #[deref(forward)]
    #[deref_mut(forward)]
    #[into_iterator(owned, ref, ref_mut)]
    #[cfg_attr(test, strategy(vec(any::<Move>(), 0..=10usize)
        .prop_map(StaticSeq::from_iter)))]
    StaticSeq<Move, { Moves::CAPACITY }>,
);

impl Moves {
    const CAPACITY: usize = 255;

    /// Pushes a move into the collection.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn push(&mut self, m: Move) {
        self.0.push(m);
    }

    /// An iterator over the [`Move`]s in this collection.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    /// A mutable iterator over the [`Move`]s in this collection.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn iter_mut(&mut self) -> <&mut Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    /// Rates all [`Move`]s in this collection.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn rate<F: FnMut(Move) -> Rating>(self, f: F) -> RatedMoves {
        let mut moves = RatedMoves {
            moves: self,
            ratings: zeroed(),
            unsorted: zeroed(),
        };

        moves.rate(f);
        moves
    }
}

impl FromIterator<Move> for Moves {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = Move>>(iter: I) -> Self {
        Moves(iter.into_iter().collect())
    }
}

/// A relative measure for how good a [`Move`] is.
pub type Rating = Bounded<i16>;

/// A collection of [`Move`]s paired with their [`Rating`]s.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deref, IntoIterator)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct RatedMoves {
    #[deref(forward)]
    #[into_iterator(owned, ref)]
    moves: Moves,

    ratings: [Rating; Moves::CAPACITY],

    // Index of the first unsorted move
    #[cfg_attr(test, strategy(Just(0)))]
    unsorted: <ConstCapacity as Capacity>::Usize,
}

impl Default for RatedMoves {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn default() -> Self {
        Self {
            moves: Default::default(),
            ratings: zeroed(),
            unsorted: zeroed(),
        }
    }
}

impl RatedMoves {
    /// An iterator over the [`RatedMove`]s in this collection in arbitrary order.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    /// An iterator over the [`Move`]s in this collection sorted by their [`Rating`]s.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn sorted(&mut self) -> SortedRatedMovesIter<'_> {
        SortedRatedMovesIter::new(self)
    }

    /// Re-rates all [`Move`]s in this collection.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn rate<F: FnMut(Move) -> Rating>(&mut self, mut f: F) {
        self.unsorted = 0;
        for (m, rating) in self.moves.iter().zip(&mut self.ratings) {
            *rating = f(*m);
        }
    }
}

/// A lazily sorted iterator of [`Move`]s.
#[derive(Debug)]
pub struct SortedRatedMovesIter<'a> {
    inner: &'a mut RatedMoves,
    cursor: <ConstCapacity as Capacity>::Usize,
}

impl<'a> SortedRatedMovesIter<'a> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn new(inner: &'a mut RatedMoves) -> Self {
        SortedRatedMovesIter { inner, cursor: 0 }
    }
}

impl Iterator for SortedRatedMovesIter<'_> {
    type Item = Move;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn next(&mut self) -> Option<Self::Item> {
        let cursor = self.cursor.cast::<usize>();
        let len = self.inner.len();
        if cursor >= len {
            return None;
        }

        if cursor >= self.inner.unsorted.cast::<usize>() {
            let ratings = self.inner.ratings.get(cursor..len).assume();

            const MASK: usize = Moves::CAPACITY.next_power_of_two() - 1;
            const SHIFT: u32 = MASK.trailing_ones();

            let mut best = i32::MIN;
            for (i, rating) in ratings.iter().enumerate() {
                best = best.max((rating.cast::<i32>() << SHIFT) | (MASK - i).cast::<i32>());
            }

            let idx = cursor + MASK - (best.cast::<usize>() & MASK);

            if cursor < idx {
                unsafe { self.inner.moves.swap_unchecked(cursor, idx) };
                unsafe { self.inner.ratings.swap_unchecked(cursor, idx) };
            }
        }

        self.cursor += 1;
        self.inner.unsorted = self.inner.unsorted.max(self.cursor);
        Some(self.inner.moves[cursor])
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for SortedRatedMovesIter<'_> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn len(&self) -> usize {
        self.inner.len() - self.cursor.cast::<usize>()
    }
}

impl FusedIterator for SortedRatedMovesIter<'_> {}

/// Trait for types that can collect [`Move`]s.
pub trait MoveCollector {
    /// The reason why collecting moves was not possible.
    type Error;

    /// Pushes one [`Move`] to the collection.
    fn collect_one(&mut self, m: Move) -> Result<(), Self::Error>;

    /// Pushes all valid moves by pieces in `indices` to `targets`.
    fn collect_attacks(
        &mut self,
        pos: &Position,
        indices: IdxSet,
        targets: Bitboard,
    ) -> Result<(), Self::Error>;

    fn collect_pawn_promotions(
        &mut self,
        pos: &Position,
        targets: Bitboard,
    ) -> Result<(), Self::Error>;

    fn collect_pawn_pushes(&mut self, pos: &Position, targets: Bitboard)
    -> Result<(), Self::Error>;
}

impl MoveCollector for Moves {
    type Error = Infallible;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn collect_one(&mut self, m: Move) -> Result<(), Self::Error> {
        Moves::push(self, m);
        Ok(())
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn collect_attacks(
        &mut self,
        pos: &Position,
        indices: IdxSet,
        targets: Bitboard,
    ) -> Result<(), Self::Error> {
        let turn = pos.turn();
        let theirs = pos.by_color(!turn);
        let squares = pos.squares()[turn].to_simd().cast::<u16>();
        let attacks = pos.pins().attacks().to_simd() & u16x64::splat(*indices);
        let targets = targets & attacks.simd_ne(zeroed());

        for wt in targets & theirs {
            let moves = squares | u16x16::splat(((wt as u16) << 6) | 0b1000000000000000);
            self.0.extend_from_simd(moves, attacks.as_array()[wt]);
        }

        for wt in targets & !theirs {
            let moves = squares | u16x16::splat((wt as u16) << 6);
            self.0.extend_from_simd(moves, attacks.as_array()[wt]);
        }

        Ok(())
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn collect_pawn_promotions(
        &mut self,
        pos: &Position,
        targets: Bitboard,
    ) -> Result<(), Self::Error> {
        const PUSH_SHIFT: [u32; Color::MAX as usize + 1] = [48, 8];
        const PUSHES: [Aligned<[Option<Move>; 32]>; Color::MAX as usize + 1] = {
            let mut table = [Aligned([None; 32]); Color::MAX as usize + 1];

            let mut i = 8;
            while i > 0 {
                i -= 1;

                let mut j = 4;
                while j > 0 {
                    j -= 1;

                    let promotion = Some([Role::Queen, Role::Rook, Role::Bishop, Role::Knight][j]);

                    let wc = Num::saturate(i + PUSH_SHIFT[Color::White] as usize);
                    table[Color::White][4 * i + j] = Some(Move::regular(wc, wc + 8, promotion));

                    let wc = Num::saturate(i + PUSH_SHIFT[Color::Black] as usize);
                    table[Color::Black][4 * i + j] = Some(Move::regular(wc, wc - 8, promotion));
                }
            }

            table
        };

        let turn = pos.turn();
        let wt = targets & pos.vacant();
        let pawns = pos.by_piece(Piece::new(Role::Pawn, turn)) & pos.pins().unpinned();

        let pushes = match turn {
            Color::White => (wt >> 8) & pawns,
            Color::Black => (wt << 8) & pawns,
        };

        let mask = (pushes >> PUSH_SHIFT[turn]).cast::<u8>();

        if mask != 0 {
            let nibbler = u8x8::from_array([1, 2, 4, 8, 16, 32, 64, 128]);
            let masked = u8x8::splat(mask) & nibbler;

            let nibbled = masked
                .simd_eq(nibbler)
                .select(u8x8::splat(0x0F), u8x8::splat(0x00));

            let lo: u8x4 = simd_swizzle!(nibbled, [0, 2, 4, 6]);
            let hi: u8x4 = simd_swizzle!(nibbled, [1, 3, 5, 7]);
            let packed = lo | (hi << u8x4::splat(4));

            let expanded_mask = u32::from_le_bytes(packed.to_array());
            let moves = PUSHES[turn].cast_ref::<u16x32>();
            self.0.extend_from_simd(*moves, expanded_mask);
        }

        let theirs = pos.by_color(!turn);
        let indices = IdxSet::from(pos.roles()[turn].matching(Some(Role::Pawn)));
        let attacks = pos.pins().attacks().to_simd() & u16x64::splat(*indices);
        for wt in targets & theirs & attacks.simd_ne(zeroed()) {
            for idx in indices & attacks.as_array()[wt] {
                let wc = pos.squares()[turn][idx].assume();
                self.collect_one(Move::capture(wc, wt, Some(Role::Knight)));
                self.collect_one(Move::capture(wc, wt, Some(Role::Bishop)));
                self.collect_one(Move::capture(wc, wt, Some(Role::Rook)));
                self.collect_one(Move::capture(wc, wt, Some(Role::Queen)));
            }
        }

        Ok(())
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn collect_pawn_pushes(
        &mut self,
        pos: &Position,
        targets: Bitboard,
    ) -> Result<(), Self::Error> {
        const SINGLE_SHIFT: u32 = 16;
        const SINGLE: [Aligned<[Option<Move>; 32]>; Color::MAX as usize + 1] = {
            let mut table = [Aligned([None; 32]); Color::MAX as usize + 1];

            let mut i = 32;
            while i > 0 {
                i -= 1;
                let wc = Num::saturate(i + SINGLE_SHIFT as usize);
                table[Color::White][i] = Some(Move::regular(wc, wc + 8, None));
                table[Color::Black][i] = Some(Move::regular(wc, wc - 8, None));
            }

            table
        };

        const DOUBLE_SHIFT: [u32; Color::MAX as usize + 1] = [8, 48];
        const DOUBLE: [Aligned<[Option<Move>; 16]>; Color::MAX as usize + 1] = {
            let mut table = [Aligned([None; 16]); Color::MAX as usize + 1];

            let mut i = 8;
            while i > 0 {
                i -= 1;

                let wc = Num::saturate(i + DOUBLE_SHIFT[Color::White] as usize);
                table[Color::White][i] = Some(Move::regular(wc, wc + 16, None));
                table[Color::White][i + 8] = Some(Move::regular(wc, wc + 8, None));

                let wc = Num::saturate(i + DOUBLE_SHIFT[Color::Black] as usize);
                table[Color::Black][i] = Some(Move::regular(wc, wc - 16, None));
                table[Color::Black][i + 8] = Some(Move::regular(wc, wc - 8, None));
            }

            table
        };

        let turn = pos.turn();
        let vacant = pos.vacant().to_bitmask();
        let wt = targets & vacant;

        let unpinned_pushes = pos.king(turn).file().bitboard() | pos.pins().unpinned();
        let pawns = unpinned_pushes & pos.by_piece(Piece::new(Role::Pawn, turn));

        let single = match turn {
            Color::White => pawns & (wt >> 8),
            Color::Black => pawns & (wt << 8),
        };

        let mask = (single >> SINGLE_SHIFT).cast::<u32>();

        if mask != 0 {
            let moves = SINGLE[turn].cast_ref::<u16x32>();
            self.0.extend_from_simd(*moves, mask);
        }

        let double = match turn {
            Color::White => pawns & (vacant >> 8) & (wt >> 16),
            Color::Black => pawns & (vacant << 8) & (wt << 16),
        };

        let mut mask = (double >> DOUBLE_SHIFT[turn]).cast::<u8>().cast::<u16>();
        mask |= (single >> DOUBLE_SHIFT[turn]).cast::<u16>() << 8;

        if mask != 0 {
            let moves = DOUBLE[turn].cast_ref::<u16x16>();
            self.0.extend_from_simd(*moves, mask);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::sample::select;
    use std::cmp::Reverse;
    use test_strategy::proptest;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn move_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Move>>(), size_of::<Move>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn decoding_encoded_move_is_an_identity(m: Move) {
        assert_eq!(Move::decode(m.encode()), m);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn capture_move_can_be_constructed(
        wc: Square,
        #[filter(#wc != #wt)] wt: Square,
        #[strategy(select(&[Role::Knight, Role::Bishop, Role::Rook, Role::Queen]))] p: Role,
    ) {
        assert!(Move::capture(wc, wt, Some(p)).is_capture());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn quiet_move_can_be_constructed(wc: Square, #[filter(#wc != #wt)] wt: Square) {
        assert!(Move::regular(wc, wt, None).is_quiet());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn promotion_move_can_be_constructed(
        wc: Square,
        #[filter(#wc != #wt)] wt: Square,
        #[strategy(select(&[Role::Knight, Role::Bishop, Role::Rook, Role::Queen]))] p: Role,
    ) {
        assert!(Move::regular(wc, wt, Some(p)).is_promotion());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn promotions_are_noisy(
        wc: Square,
        #[filter(#wc != #wt)] wt: Square,
        #[strategy(select(&[Role::Knight, Role::Bishop, Role::Rook, Role::Queen]))] p: Role,
    ) {
        assert!(Move::regular(wc, wt, Some(p)).is_noisy());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn captures_are_noisy(
        wc: Square,
        #[filter(#wc != #wt)] wt: Square,
        #[strategy(select(&[Role::Knight, Role::Bishop, Role::Rook, Role::Queen]))] p: Role,
    ) {
        assert!(Move::capture(wc, wt, None).is_noisy());
        assert!(Move::capture(wc, wt, Some(p)).is_noisy());
    }

    #[proptest]
    fn sorted_is_deterministic(mut ms: RatedMoves) {
        assert_eq!(Vec::from_iter(ms.sorted()), Vec::from_iter(ms.sorted()));
    }

    #[proptest]
    fn sorted_iterates_through_moves_by_highest_rating(mut ms: RatedMoves) {
        let mut rms = Vec::from_iter(ms.iter().zip(&ms.ratings));
        rms.sort_by_key(|(_, r)| Reverse(**r));

        assert_eq!(
            Vec::from_iter(rms.into_iter().map(|(m, _)| *m)),
            Vec::from_iter(ms.sorted()),
        );
    }

    #[proptest]
    fn rate_defines_move_order(mut ms: RatedMoves) {
        let mut rating = Rating::new(0);

        ms.rate(|_| {
            rating += 1;
            rating
        });

        let mut ns = Vec::from_iter(ms.clone());
        ns.reverse();

        assert_eq!(Vec::from_iter(ms.sorted()), ns);
    }
}

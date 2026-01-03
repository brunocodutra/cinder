use crate::chess::{Bitboard, Color, Flip, Magic, Perspective, Rank, Role, Square};
use crate::util::{Assume, Int};
use bytemuck::zeroed;
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::{cell::SyncUnsafeCell, ops::Shl, str::FromStr};

/// A chess [piece][`Role`] of a certain [`Color`].
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq, Ord, PartialOrd)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(u8)]
pub enum Piece {
    WhitePawn,
    BlackPawn,
    WhiteKnight,
    BlackKnight,
    WhiteBishop,
    BlackBishop,
    WhiteRook,
    BlackRook,
    WhiteQueen,
    BlackQueen,
    WhiteKing,
    BlackKing,
}

impl Piece {
    #[inline(always)]
    const fn forks(wc: Square, color: Color) -> Bitboard {
        pub static FORKS: SyncUnsafeCell<[[Bitboard; 64]; 2]> = SyncUnsafeCell::new(zeroed());

        #[cold]
        #[ctor::ctor]
        #[cfg(not(miri))]
        #[inline(never)]
        unsafe fn init() {
            let forks = unsafe { FORKS.get().as_mut_unchecked() };

            for color in Color::iter() {
                for wc in Square::iter() {
                    let steps = [(-1, 1), (1, 1)];
                    let moves = Bitboard::fill(wc.perspective(color), &steps, Bitboard::full());
                    forks[color as usize][wc as usize] = moves.perspective(color).without(wc);
                }
            }
        }

        unsafe { FORKS.get().as_ref_unchecked()[color as usize][wc as usize] }
    }

    #[inline(always)]
    const fn jumps(wc: Square) -> Bitboard {
        pub static JUMPS: SyncUnsafeCell<[Bitboard; 64]> = SyncUnsafeCell::new(zeroed());

        #[cold]
        #[ctor::ctor]
        #[cfg(not(miri))]
        #[inline(never)]
        unsafe fn init() {
            let jumps = unsafe { JUMPS.get().as_mut_unchecked() };

            for wc in Square::iter() {
                #[rustfmt::skip]
                let steps = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)];
                let moves = Bitboard::fill(wc, &steps, Bitboard::full()).without(wc);
                jumps[wc as usize] = moves;
            }
        }

        unsafe { JUMPS.get().as_ref_unchecked()[wc as usize] }
    }

    #[inline(always)]
    const fn steps(wc: Square) -> Bitboard {
        pub static SLIDES: SyncUnsafeCell<[Bitboard; 64]> = SyncUnsafeCell::new(zeroed());

        #[cold]
        #[ctor::ctor]
        #[cfg(not(miri))]
        #[inline(never)]
        unsafe fn init() {
            let slides = unsafe { SLIDES.get().as_mut_unchecked() };

            for wc in Square::iter() {
                #[rustfmt::skip]
                let steps = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];
                let moves = Bitboard::fill(wc, &steps, Bitboard::full()).without(wc);
                slides[wc as usize] = moves;
            }
        }

        unsafe { SLIDES.get().as_ref_unchecked()[wc as usize] }
    }

    #[inline(always)]
    const fn slides(idx: usize) -> Bitboard {
        pub static BITBOARDS: SyncUnsafeCell<[Bitboard; 88772]> = SyncUnsafeCell::new(zeroed());

        #[cold]
        #[ctor::ctor]
        #[cfg(not(miri))]
        #[inline(never)]
        unsafe fn init() {
            let bitboard = unsafe { BITBOARDS.get().as_mut_unchecked() };

            for wc in Square::iter() {
                let magic = Magic::bishop(wc);
                for bb in magic.mask().subsets() {
                    let blockers = bb | !magic.mask();
                    let steps = [(-1, 1), (1, 1), (1, -1), (-1, -1)];
                    let moves = Bitboard::fill(wc, &steps, blockers).without(wc);
                    let idx = (bb.wrapping_mul(magic.factor()) >> 55) as usize + magic.offset();
                    debug_assert!(bitboard[idx] == moves || bitboard[idx] == Bitboard::empty());
                    bitboard[idx] = moves;
                }

                let magic = Magic::rook(wc);
                for bb in magic.mask().subsets() {
                    let blockers = bb | !magic.mask();
                    let steps = [(-1, 0), (0, 1), (1, 0), (0, -1)];
                    let moves = Bitboard::fill(wc, &steps, blockers).without(wc);
                    let idx = (bb.wrapping_mul(magic.factor()) >> 52) as usize + magic.offset();
                    debug_assert!(bitboard[idx] == moves || bitboard[idx] == Bitboard::empty());
                    bitboard[idx] = moves;
                }
            }
        }

        unsafe { *BITBOARDS.get().as_ref_unchecked().get(idx).assume() }
    }

    /// Constructs [`Piece`] from a pair of [`Color`] and [`Role`].
    #[inline(always)]
    pub const fn new(r: Role, c: Color) -> Self {
        Int::new(c.get() | (r.get() << 1))
    }

    /// This piece's [`Role`].
    #[inline(always)]
    pub const fn role(&self) -> Role {
        Int::new(self.get() >> 1)
    }

    /// This piece's [`Color`].
    #[inline(always)]
    pub const fn color(&self) -> Color {
        Int::new(self.get() & 0b1)
    }

    /// This piece's possible attacks from a given square.
    #[inline(always)]
    pub const fn attacks(&self, wc: Square, occupied: Bitboard) -> Bitboard {
        match self.role() {
            Role::Pawn => Self::forks(wc, self.color()),
            Role::Knight => Self::jumps(wc),
            Role::King => Self::steps(wc),

            Role::Bishop => {
                let magic = Magic::bishop(wc);
                let blockers = occupied & magic.mask();
                let idx = (blockers.wrapping_mul(magic.factor()) >> 55) as usize + magic.offset();
                Self::slides(idx)
            }

            Role::Rook => {
                let magic = Magic::rook(wc);
                let blockers = occupied & magic.mask();
                let idx = (blockers.wrapping_mul(magic.factor()) >> 52) as usize + magic.offset();
                Self::slides(idx)
            }

            Role::Queen => {
                let magic = Magic::bishop(wc);
                let blockers = occupied & magic.mask();
                let idb = (blockers.wrapping_mul(magic.factor()) >> 55) as usize + magic.offset();
                let magic = Magic::rook(wc);
                let blockers = occupied & magic.mask();
                let idr = (blockers.wrapping_mul(magic.factor()) >> 52) as usize + magic.offset();
                Self::slides(idb) | Self::slides(idr)
            }
        }
    }

    /// This piece's possible moves from a given square.
    #[inline(always)]
    pub const fn moves(&self, wc: Square, ours: Bitboard, theirs: Bitboard) -> Bitboard {
        let occ = ours ^ theirs;
        if self.role() != Role::Pawn {
            self.attacks(wc, occ) & !ours
        } else {
            let empty = !occ;
            let color = self.color();
            let third = Rank::Third.bitboard();
            let push = wc.bitboard().perspective(color).shl(8).perspective(color) & empty;
            push | ((push.perspective(color) & third).shl(8).perspective(color) & empty)
        }
    }
}

unsafe impl const Int for Piece {
    type Repr = u8;
    const MIN: Self::Repr = Piece::WhitePawn as u8;
    const MAX: Self::Repr = Piece::BlackKing as u8;
}

impl const Flip for Piece {
    /// Mirrors this piece's [`Color`].
    #[inline(always)]
    fn flip(self) -> Self {
        Int::new(self.get() ^ Piece::BlackPawn.get())
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Piece::WhitePawn => f.write_char('P'),
            Piece::BlackPawn => f.write_char('p'),
            Piece::WhiteKnight => f.write_char('N'),
            Piece::BlackKnight => f.write_char('n'),
            Piece::WhiteBishop => f.write_char('B'),
            Piece::BlackBishop => f.write_char('b'),
            Piece::WhiteRook => f.write_char('R'),
            Piece::BlackRook => f.write_char('r'),
            Piece::WhiteQueen => f.write_char('Q'),
            Piece::BlackQueen => f.write_char('q'),
            Piece::WhiteKing => f.write_char('K'),
            Piece::BlackKing => f.write_char('k'),
        }
    }
}

/// The reason why parsing [`Piece`] failed.
#[derive(Debug, Display, Error)]
#[derive_const(Default, Clone, Eq, PartialEq)]
#[display("failed to parse piece")]
pub struct ParsePieceError;

impl FromStr for Piece {
    type Err = ParsePieceError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "P" => Ok(Piece::WhitePawn),
            "p" => Ok(Piece::BlackPawn),
            "N" => Ok(Piece::WhiteKnight),
            "n" => Ok(Piece::BlackKnight),
            "B" => Ok(Piece::WhiteBishop),
            "b" => Ok(Piece::BlackBishop),
            "R" => Ok(Piece::WhiteRook),
            "r" => Ok(Piece::BlackRook),
            "Q" => Ok(Piece::WhiteQueen),
            "q" => Ok(Piece::BlackQueen),
            "K" => Ok(Piece::WhiteKing),
            "k" => Ok(Piece::BlackKing),
            _ => Err(ParsePieceError),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn piece_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Piece>>(), size_of::<Piece>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn piece_has_a_color(r: Role, c: Color) {
        assert_eq!(Piece::new(r, c).color(), c);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn piece_has_a_role(r: Role, c: Color) {
        assert_eq!(Piece::new(r, c).role(), r);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn piece_cannot_attack_onto_themselves(p: Piece, wc: Square, bb: Bitboard) {
        assert!(!p.attacks(wc, bb).contains(wc));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn piece_cannot_move_onto_themselves(p: Piece, wc: Square, a: Bitboard, b: Bitboard) {
        assert!(!p.moves(wc, a, b).contains(wc));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn piece_can_only_move_to_empty_or_opponent_piece(
        p: Piece,
        wc: Square,
        a: Bitboard,
        b: Bitboard,
    ) {
        for sq in p.moves(wc, a, b) {
            assert!(a.inverse().union(b).contains(sq));
        }
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn flipping_piece_preserves_role_and_mirrors_color(p: Piece) {
        assert_eq!(p.flip().role(), p.role());
        assert_eq!(p.flip().color(), !p.color());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_piece_is_an_identity(p: Piece) {
        assert_eq!(p.to_string().parse(), Ok(p));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_piece_fails_if_not_one_of_pnbrqk(
        #[filter(!['p', 'n', 'b', 'r', 'q', 'k'].contains(&#c.to_ascii_lowercase()))] c: char,
    ) {
        assert_eq!(c.to_string().parse::<Piece>(), Err(ParsePieceError));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_piece_fails_if_length_not_one(#[filter(#s.len() != 1)] s: String) {
        assert_eq!(s.parse::<Piece>(), Err(ParsePieceError));
    }
}

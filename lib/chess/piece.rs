use crate::chess::{Bitboard, Color, Magic, Perspective, Role, Square};
use crate::util::Integer;
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::{cell::SyncUnsafeCell, mem::MaybeUninit, str::FromStr};

/// A chess [piece][`Role`] of a certain [`Color`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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
    fn bitboard(idx: usize) -> Bitboard {
        static BITBOARDS: SyncUnsafeCell<[Bitboard; 88772]> =
            unsafe { MaybeUninit::zeroed().assume_init() };

        #[cold]
        #[ctor::ctor]
        #[inline(never)]
        unsafe fn init() {
            let bitboard = BITBOARDS.get().as_mut_unchecked();

            for whence in Square::iter() {
                let (pushes, attacks) = Magic::pawn(whence);
                for bb in pushes.mask().subsets() {
                    let blks = bb | !pushes.mask();
                    let moves = Bitboard::fill(whence, &[(0, 1)], blks).without(whence) & !blks;
                    let idx = (bb.wrapping_mul(pushes.factor()) >> 62) as usize + pushes.offset();
                    debug_assert!(bitboard[idx] == moves || bitboard[idx] == Bitboard::empty());
                    bitboard[idx] = moves;
                }

                let steps = [(-1, 1), (1, 1)];
                let idx = attacks.offset();
                let moves = Bitboard::fill(whence, &steps, Bitboard::full()).without(whence);
                debug_assert!(bitboard[idx] == moves || bitboard[idx] == Bitboard::empty());
                bitboard[idx] = moves;

                let magic = Magic::knight(whence);
                #[rustfmt::skip]
                let steps = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)];
                let moves = Bitboard::fill(whence, &steps, Bitboard::full()).without(whence);
                let idx = magic.offset();
                debug_assert!(bitboard[idx] == moves || bitboard[idx] == Bitboard::empty());
                bitboard[idx] = moves;

                let magic = Magic::bishop(whence);
                for bb in magic.mask().subsets() {
                    let blks = bb | !magic.mask();
                    let steps = [(-1, 1), (1, 1), (1, -1), (-1, -1)];
                    let moves = Bitboard::fill(whence, &steps, blks).without(whence);
                    let idx = (bb.wrapping_mul(magic.factor()) >> 55) as usize + magic.offset();
                    debug_assert!(bitboard[idx] == moves || bitboard[idx] == Bitboard::empty());
                    bitboard[idx] = moves;
                }

                let magic = Magic::rook(whence);
                for bb in magic.mask().subsets() {
                    let blks = bb | !magic.mask();
                    let steps = [(-1, 0), (0, 1), (1, 0), (0, -1)];
                    let moves = Bitboard::fill(whence, &steps, blks).without(whence);
                    let idx = (bb.wrapping_mul(magic.factor()) >> 52) as usize + magic.offset();
                    debug_assert!(bitboard[idx] == moves || bitboard[idx] == Bitboard::empty());
                    bitboard[idx] = moves;
                }

                let magic = Magic::king(whence);
                #[rustfmt::skip]
                let steps = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)];
                let moves = Bitboard::fill(whence, &steps, Bitboard::full()).without(whence);
                let idx = magic.offset();
                debug_assert!(bitboard[idx] == moves || bitboard[idx] == Bitboard::empty());
                bitboard[idx] = moves;
            }
        }

        *unsafe { BITBOARDS.get().as_ref_unchecked().get_unchecked(idx) }
    }

    /// Constructs [`Piece`] from a pair of [`Color`] and [`Role`].
    #[inline(always)]
    pub fn new(r: Role, c: Color) -> Self {
        <Self as Integer>::new(c.get() | (r.get() << 1))
    }

    /// This piece's [`Role`].
    #[inline(always)]
    pub fn role(&self) -> Role {
        Role::new(self.get() >> 1)
    }

    /// This piece's [`Color`].
    #[inline(always)]
    pub fn color(&self) -> Color {
        Color::new(self.get() & 0b1)
    }

    /// This piece's possible target squares from a given square.
    #[inline(always)]
    pub fn targets(&self, whence: Square) -> Bitboard {
        self.attacks(whence, Bitboard::empty())
    }

    /// This piece's possible attacks from a given square.
    #[inline(always)]
    pub fn attacks(&self, whence: Square, blockers: Bitboard) -> Bitboard {
        match self.role() {
            Role::Pawn => {
                let color = self.color();
                let (_, magic) = Magic::pawn(whence.perspective(color));
                Self::bitboard(magic.offset()).perspective(color)
            }

            Role::Knight => Self::bitboard(Magic::knight(whence).offset()),

            Role::Bishop => {
                let magic = Magic::bishop(whence);
                let blks = blockers & magic.mask();
                let idx = (blks.wrapping_mul(magic.factor()) >> 55) as usize + magic.offset();
                Self::bitboard(idx)
            }

            Role::Rook => {
                let magic = Magic::rook(whence);
                let blks = blockers & magic.mask();
                let idx = (blks.wrapping_mul(magic.factor()) >> 52) as usize + magic.offset();
                Self::bitboard(idx)
            }

            Role::Queen => {
                let magic = Magic::bishop(whence);
                let blks = blockers & magic.mask();
                let bishop = (blks.wrapping_mul(magic.factor()) >> 55) as usize + magic.offset();
                let magic = Magic::rook(whence);
                let blks = blockers & magic.mask();
                let rook = (blks.wrapping_mul(magic.factor()) >> 52) as usize + magic.offset();
                Self::bitboard(bishop) | Self::bitboard(rook)
            }

            Role::King => Self::bitboard(Magic::king(whence).offset()),
        }
    }

    /// This piece's possible moves from a given square.
    #[inline(always)]
    pub fn moves(&self, whence: Square, ours: Bitboard, theirs: Bitboard) -> Bitboard {
        let blockers = ours | theirs;

        if self.role() != Role::Pawn {
            self.attacks(whence, blockers) & !ours
        } else {
            let color = self.color();
            let (magic, _) = Magic::pawn(whence.perspective(color));
            let blks = blockers.perspective(color) & magic.mask();
            let idx = (blks.wrapping_mul(magic.factor()) >> 62) as usize + magic.offset();
            Self::bitboard(idx).perspective(color)
        }
    }
}

unsafe impl Integer for Piece {
    type Repr = u8;
    const MIN: Self::Repr = Piece::WhitePawn as _;
    const MAX: Self::Repr = Piece::BlackKing as _;
}

impl Perspective for Piece {
    /// Mirrors this piece's [`Color`].
    #[inline(always)]
    fn flip(&self) -> Self {
        <Self as Integer>::new(self.get() ^ Piece::BlackPawn.get())
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

/// The reason why parsing the piece.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
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
    use std::mem::size_of;
    use test_strategy::proptest;

    #[test]
    fn piece_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Piece>>(), size_of::<Piece>());
    }

    #[proptest]
    fn piece_has_a_color(r: Role, c: Color) {
        assert_eq!(Piece::new(r, c).color(), c);
    }

    #[proptest]
    fn piece_has_a_role(r: Role, c: Color) {
        assert_eq!(Piece::new(r, c).role(), r);
    }

    #[proptest]
    fn piece_cannot_attack_onto_themselves(p: Piece, wc: Square, bb: Bitboard) {
        assert!(!p.attacks(wc, bb).contains(wc));
    }

    #[proptest]
    fn piece_cannot_move_onto_themselves(p: Piece, wc: Square, a: Bitboard, b: Bitboard) {
        assert!(!p.moves(wc, a, b).contains(wc));
    }

    #[proptest]
    fn piece_can_only_move_to_empty_or_opponent_piece(
        p: Piece,
        wc: Square,
        a: Bitboard,
        b: Bitboard,
    ) {
        for sq in p.moves(wc, a, b) {
            assert!(a.inverse().union(b).contains(sq))
        }
    }

    #[proptest]
    fn flipping_piece_preserves_role_and_mirrors_color(p: Piece) {
        assert_eq!(p.flip().role(), p.role());
        assert_eq!(p.flip().color(), !p.color());
    }

    #[proptest]
    fn parsing_printed_piece_is_an_identity(p: Piece) {
        assert_eq!(p.to_string().parse(), Ok(p));
    }

    #[proptest]
    fn parsing_piece_fails_if_not_one_of_pnbrqk(
        #[filter(!['p', 'n', 'b', 'r', 'q', 'k'].contains(&#c.to_ascii_lowercase()))] c: char,
    ) {
        assert_eq!(c.to_string().parse::<Piece>(), Err(ParsePieceError));
    }

    #[proptest]
    fn parsing_piece_fails_if_length_not_one(#[filter(#s.len() != 1)] s: String) {
        assert_eq!(s.parse::<Piece>(), Err(ParsePieceError));
    }
}

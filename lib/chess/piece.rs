use crate::chess::{Bitboard, Color, Flip, Perspective, Rank, Role, Square};
use crate::simd::*;
use crate::util::{Assume, Binary, Bits, Int, Niched, Num};
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::ops::{Index, IndexMut};
use std::str::FromStr;

/// A chess [piece][`Role`] of a certain [`Color`].
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq, PartialOrd, Ord)]
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

const unsafe impl Num for Piece {
    type Repr = u8;
    const MIN: Self::Repr = Piece::WhitePawn as u8;
    const MAX: Self::Repr = Piece::BlackKing as u8;
}

const unsafe impl Int for Piece {}

const unsafe impl Niched for Piece {}

const impl Piece {
    #[expect(dead_code)]
    const REQUIRES: () = const { assert!(size_of::<Self>() == size_of::<Option<Self>>()) };

    pub const LEN: usize = Self::MAX as usize + 1;

    pub const ENCODER: u8x64 = const {
        let mut encoder = [0xFF; 16];

        use Piece::*;
        encoder[WhitePawn as usize] = Role::ENCODER.as_array()[Role::Pawn as usize];
        encoder[WhiteKnight as usize] = Role::ENCODER.as_array()[Role::Knight as usize];
        encoder[WhiteBishop as usize] = Role::ENCODER.as_array()[Role::Bishop as usize];
        encoder[WhiteRook as usize] = Role::ENCODER.as_array()[Role::Rook as usize];
        encoder[WhiteQueen as usize] = Role::ENCODER.as_array()[Role::Queen as usize];
        encoder[WhiteKing as usize] = Role::ENCODER.as_array()[Role::King as usize];
        encoder[BlackPawn as usize] = Role::ENCODER.as_array()[Role::Pawn as usize] | 0b1000;
        encoder[BlackKnight as usize] = Role::ENCODER.as_array()[Role::Knight as usize] | 0b1000;
        encoder[BlackBishop as usize] = Role::ENCODER.as_array()[Role::Bishop as usize] | 0b1000;
        encoder[BlackRook as usize] = Role::ENCODER.as_array()[Role::Rook as usize] | 0b1000;
        encoder[BlackQueen as usize] = Role::ENCODER.as_array()[Role::Queen as usize] | 0b1000;
        encoder[BlackKing as usize] = Role::ENCODER.as_array()[Role::King as usize] | 0b1000;
        Aligned([encoder; 4]).cast()
    };

    pub const DECODER: u8x64 = const {
        let mut decoder = [0xFF; 16];

        use Piece::*;
        decoder[Piece::ENCODER.as_array()[WhitePawn as usize] as usize] = WhitePawn.get();
        decoder[Piece::ENCODER.as_array()[WhiteKnight as usize] as usize] = WhiteKnight.get();
        decoder[Piece::ENCODER.as_array()[WhiteBishop as usize] as usize] = WhiteBishop.get();
        decoder[Piece::ENCODER.as_array()[WhiteRook as usize] as usize] = WhiteRook.get();
        decoder[Piece::ENCODER.as_array()[WhiteQueen as usize] as usize] = WhiteQueen.get();
        decoder[Piece::ENCODER.as_array()[WhiteKing as usize] as usize] = WhiteKing.get();
        decoder[Piece::ENCODER.as_array()[BlackPawn as usize] as usize] = BlackPawn.get();
        decoder[Piece::ENCODER.as_array()[BlackKnight as usize] as usize] = BlackKnight.get();
        decoder[Piece::ENCODER.as_array()[BlackBishop as usize] as usize] = BlackBishop.get();
        decoder[Piece::ENCODER.as_array()[BlackRook as usize] as usize] = BlackRook.get();
        decoder[Piece::ENCODER.as_array()[BlackQueen as usize] as usize] = BlackQueen.get();
        decoder[Piece::ENCODER.as_array()[BlackKing as usize] as usize] = BlackKing.get();
        Aligned([decoder; 4]).cast()
    };

    /// Constructs [`Piece`] from a pair of [`Color`] and [`Role`].
    #[inline(always)]
    pub fn new(r: Role, c: Color) -> Self {
        Num::new(c.get() | (r.get() << 1))
    }

    /// This piece's [`Role`].
    #[inline(always)]
    pub fn role(self) -> Role {
        Num::new(self.get() >> 1)
    }

    /// This piece's [`Color`].
    #[inline(always)]
    pub fn color(self) -> Color {
        Num::new(self.get() & 0b1)
    }

    /// This piece's representation in ASCII.
    #[inline(always)]
    pub fn to_ascii(self) -> u8 {
        match self.color() {
            Color::White => self.role().to_ascii().to_ascii_uppercase(),
            Color::Black => self.role().to_ascii(),
        }
    }

    /// A [`Bitboard`] for this piece's attack pattern from a [`Square`].
    #[inline(always)]
    pub fn attacks(self, sq: Square) -> Bitboard {
        const ATTACKS: [[Bitboard; Square::LEN]; 7] = const {
            let mut table = [[Bitboard::empty(); Square::LEN]; 7];

            for color in Color::iter() {
                for wc in Square::iter() {
                    if (Rank::Second..=Rank::Seventh).contains(&wc.rank()) {
                        let steps = [(-1, 1), (1, 1)];
                        let moves = Bitboard::fill(wc.perspective(color), &steps, Bitboard::full());
                        table[color as usize][wc] |= moves.perspective(color).without(wc);
                    }
                }
            }

            for wc in Square::iter() {
                #[rustfmt::skip]
                let steps = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
                let moves = Bitboard::fill(wc, &steps, Bitboard::full()).without(wc);
                table[Role::Knight as usize + 1][wc] |= moves;
            }

            for wc in Square::iter() {
                #[rustfmt::skip]
                let steps = [(1, 1), (1, -1), (-1, -1), (-1, 1)];
                let moves = Bitboard::fill(wc, &steps, Bitboard::empty()).without(wc);
                table[Role::Bishop as usize + 1][wc] |= moves;
                table[Role::Queen as usize + 1][wc] |= moves;
            }

            for wc in Square::iter() {
                #[rustfmt::skip]
                let steps = [(0, 1), (1, 0), (0, -1), (-1, 0)];
                let moves = Bitboard::fill(wc, &steps, Bitboard::empty()).without(wc);
                table[Role::Rook as usize + 1][wc] |= moves;
                table[Role::Queen as usize + 1][wc] |= moves;
            }

            for wc in Square::iter() {
                #[rustfmt::skip]
                let steps = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
                let moves = Bitboard::fill(wc, &steps, Bitboard::full()).without(wc);
                table[Role::King as usize + 1][wc] |= moves;
            }

            table
        };

        match self {
            Piece::WhitePawn => ATTACKS[0][sq],
            Piece::BlackPawn => ATTACKS[1][sq],
            _ => ATTACKS[self.role() as usize + 1][sq],
        }
    }
}

const impl Flip for Piece {
    /// Mirrors this piece's [`Color`].
    #[inline(always)]
    fn flip(self) -> Self {
        Num::new(self.get() ^ Piece::BlackPawn.get())
    }
}

const impl Binary for Piece {
    type Bits = Bits<u8, 4>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        let encoded = Self::ENCODER.as_array()[self.cast::<usize>()];
        encoded.convert().assume()
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        let decoded = Self::DECODER.as_array()[bits.cast::<usize>()];
        decoded.convert().assume()
    }
}

const impl<T> Index<Piece> for [T; Piece::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, p: Piece) -> &Self::Output {
        self.get(p.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Piece> for [T; Piece::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, p: Piece) -> &mut Self::Output {
        self.get_mut(p.cast::<usize>()).assume()
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char(self.to_ascii() as char)
    }
}

/// The reason why parsing [`Piece`] failed.
#[derive(Debug, Display, Copy, Error)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[display("failed to parse piece")]
pub struct ParsePieceError;

impl FromStr for Piece {
    type Err = ParsePieceError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let &[b] = s.as_bytes() else {
            return Err(ParsePieceError);
        };

        let s = [b.to_ascii_lowercase()];
        let Ok(r) = str::from_utf8(&s).assume().parse::<Role>() else {
            return Err(ParsePieceError);
        };

        Ok(Piece::new(r, Color::from(b.is_ascii_lowercase())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

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
    fn flipping_piece_preserves_role_and_mirrors_color(p: Piece) {
        assert_eq!(p.flip().role(), p.role());
        assert_eq!(p.flip().color(), !p.color());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn decoding_encoded_piece_is_an_identity(p: Piece) {
        assert_eq!(Piece::decode(p.encode()), p);
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

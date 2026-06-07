use crate::chess::{Color, Flip, Role};
use crate::util::{Assume, Binary, Bits, Int, Num};
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::ops::{Index, IndexMut};
use std::{hint::unreachable_unchecked, str::FromStr};

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

const impl Piece {
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
}

const unsafe impl Num for Piece {
    type Repr = u8;
    const MIN: Self::Repr = Piece::WhitePawn as u8;
    const MAX: Self::Repr = Piece::BlackKing as u8;
}

const unsafe impl Int for Piece {}

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
        match self {
            Piece::WhitePawn => Bits::new(0b0010),
            Piece::WhiteKnight => Bits::new(0b0011),
            Piece::WhiteBishop => Bits::new(0b0101),
            Piece::WhiteRook => Bits::new(0b0110),
            Piece::WhiteQueen => Bits::new(0b0111),
            Piece::WhiteKing => Bits::new(0b0001),
            Piece::BlackPawn => Bits::new(0b1010),
            Piece::BlackKnight => Bits::new(0b1011),
            Piece::BlackBishop => Bits::new(0b1101),
            Piece::BlackRook => Bits::new(0b1110),
            Piece::BlackQueen => Bits::new(0b1111),
            Piece::BlackKing => Bits::new(0b1001),
        }
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        match bits.get() {
            0b0010 => Piece::WhitePawn,
            0b0011 => Piece::WhiteKnight,
            0b0101 => Piece::WhiteBishop,
            0b0110 => Piece::WhiteRook,
            0b0111 => Piece::WhiteQueen,
            0b0001 => Piece::WhiteKing,
            0b1010 => Piece::BlackPawn,
            0b1011 => Piece::BlackKnight,
            0b1101 => Piece::BlackBishop,
            0b1110 => Piece::BlackRook,
            0b1111 => Piece::BlackQueen,
            0b1001 => Piece::BlackKing,
            _ => unsafe { unreachable_unchecked() },
        }
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
#[derive(Debug, Display, Copy, Error)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[display("failed to parse piece")]
pub struct ParsePieceError;

const impl FromStr for Piece {
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

const impl<T> Index<Piece> for [T; Piece::MAX as usize + 1] {
    type Output = T;

    #[inline(always)]
    fn index(&self, p: Piece) -> &Self::Output {
        self.get(p.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Piece> for [T; Piece::MAX as usize + 1] {
    #[inline(always)]
    fn index_mut(&mut self, p: Piece) -> &mut Self::Output {
        self.get_mut(p.cast::<usize>()).assume()
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

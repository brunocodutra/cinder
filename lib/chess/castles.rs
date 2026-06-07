use crate::chess::{Color, Move, Perspective, Piece, Role, Square};
use crate::util::{Assume, Bits, Int, Num};
use bytemuck::Zeroable;
use derive_more::with_trait::{Debug, Display, Error};
use std::fmt::{self, Formatter};
use std::{ascii::Char, ops::*, slice::Iter, str::FromStr};

/// The castling rights in a chess [`Position`][`crate::chess::Position`].
#[derive(Debug, Copy, Hash, Zeroable)]
#[derive_const(Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Castles({self})")]
pub struct Castles(Bits<u8, 4>);

const unsafe impl Num for Castles {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 15;
}

const unsafe impl Int for Castles {}

const impl Castles {
    /// No castling rights.
    #[inline(always)]
    pub fn none() -> Self {
        Castles(Bits::new(0b0000))
    }

    /// All castling rights.
    #[inline(always)]
    pub fn all() -> Self {
        Castles(Bits::new(0b1111))
    }

    /// The rook's [`Move`] given the king's castling [`Square`].
    #[inline(always)]
    pub fn rook(castling: Square) -> Option<Move> {
        match castling {
            Square::C1 => Some(Move::regular(Square::A1, Square::D1, None)),
            Square::G1 => Some(Move::regular(Square::H1, Square::F1, None)),
            Square::C8 => Some(Move::regular(Square::A8, Square::D8, None)),
            Square::G8 => Some(Move::regular(Square::H8, Square::F8, None)),
            _ => None,
        }
    }

    /// Whether the rights for the given castling square.
    #[inline(always)]
    pub fn has(self, sq: Square) -> bool {
        self & Castles::from(Castles::rook(sq).assume().whence()) != Castles::none()
    }
}

const impl Default for Castles {
    #[inline(always)]
    fn default() -> Self {
        Castles::all()
    }
}

const impl Not for Castles {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(self.0.not())
    }
}

const impl BitAnd for Castles {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0.bitand(rhs.0))
    }
}

const impl BitAndAssign for Castles {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0.bitand_assign(rhs.0);
    }
}

const impl BitOr for Castles {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0.bitor(rhs.0))
    }
}

const impl BitOrAssign for Castles {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0.bitor_assign(rhs.0);
    }
}

const impl BitXor for Castles {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0.bitxor(rhs.0))
    }
}

const impl BitXorAssign for Castles {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.0.bitxor_assign(rhs.0);
    }
}

const impl From<Square> for Castles {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        match sq {
            Square::A1 => Castles(Bits::new(0b0010)),
            Square::H1 => Castles(Bits::new(0b0001)),
            Square::E1 => Castles(Bits::new(0b0011)),
            Square::A8 => Castles(Bits::new(0b1000)),
            Square::H8 => Castles(Bits::new(0b0100)),
            Square::E8 => Castles(Bits::new(0b1100)),
            _ => Castles::none(),
        }
    }
}

impl Display for Castles {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for side in Color::iter() {
            if self.has(Square::G1.perspective(side)) {
                Display::fmt(&Piece::new(Role::King, side), f)?;
            }

            if self.has(Square::C1.perspective(side)) {
                Display::fmt(&Piece::new(Role::Queen, side), f)?;
            }
        }

        Ok(())
    }
}

const impl<T> Index<Castles> for [T; Castles::MAX as usize + 1] {
    type Output = T;

    #[inline(always)]
    fn index(&self, c: Castles) -> &Self::Output {
        self.get(c.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Castles> for [T; Castles::MAX as usize + 1] {
    #[inline(always)]
    fn index_mut(&mut self, c: Castles) -> &mut Self::Output {
        self.get_mut(c.cast::<usize>()).assume()
    }
}

/// The reason why parsing [`Castles`] failed.
#[derive(Debug, Display, Copy, Error)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[display("failed to parse castling rights")]
pub struct ParseCastlesError;

const impl FromStr for Castles
where
    for<'a> &'a [Char]: [const] IntoIterator<IntoIter = Iter<'a, Char>>,
    for<'a> Iter<'a, Char>: [const] Iterator<Item = &'a Char>,
{
    type Err = ParseCastlesError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut castles = Castles::none();

        use {Piece::*, Square::*};
        for c in s.as_ascii().ok_or(ParseCastlesError)? {
            match Piece::from_str(c.as_str()) {
                Ok(p @ (WhiteKing | BlackKing)) if !castles.has(G1.perspective(p.color())) => {
                    castles |= Castles::from(Square::H1.perspective(p.color()));
                }

                Ok(p @ (WhiteQueen | BlackQueen)) if !castles.has(C1.perspective(p.color())) => {
                    castles |= Castles::from(Square::A1.perspective(p.color()));
                }

                _ => return Err(ParseCastlesError),
            }
        }

        Ok(castles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_castles_is_an_identity(cr: Castles) {
        assert_eq!(cr.to_string().parse(), Ok(cr));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_castles_fails_if_right_is_duplicated(
        #[filter(!#s.is_empty())]
        #[strategy("(KK)?(kk)?(QQ)?(qq)?")]
        s: String,
    ) {
        assert_eq!(Castles::from_str(&s), Err(ParseCastlesError));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    #[expect(clippy::string_slice)]
    fn parsing_castles_fails_for_invalid_string(
        c: Castles,
        #[strategy(..=#c.to_string().len())] n: usize,
        #[strategy("[^[:ascii:]]+")] r: String,
    ) {
        let s = c.to_string();

        assert_eq!(
            [&s[..n], &r, &s[n..]].concat().parse().ok(),
            None::<Castles>
        );
    }
}

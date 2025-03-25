use crate::chess::{Color, Perspective, Piece, Role, Square};
use crate::util::{Bits, Integer};
use derive_more::with_trait::{Debug, *};
use std::fmt::{self, Formatter};
use std::{cell::SyncUnsafeCell, mem::MaybeUninit, str::FromStr};

/// The castling rights in a chess [`Position`][`crate::chess::Position`].
#[derive(
    Debug,
    Copy,
    Clone,
    Eq,
    PartialEq,
    Hash,
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
    BitXor,
    BitXorAssign,
    Not,
)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Castles({self})")]
pub struct Castles(Bits<u8, 4>);

impl Castles {
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

    /// A unique number the represents this castling rights configuration.
    #[inline(always)]
    pub fn index(&self) -> u8 {
        self.0.get()
    }

    /// Whether the given side has kingside castling rights.
    #[inline(always)]
    pub fn has_short(&self, side: Color) -> bool {
        *self & Castles::from(Square::H1.perspective(side)) != Castles::none()
    }

    /// Whether the given side has queenside castling rights.
    #[inline(always)]
    pub fn has_long(&self, side: Color) -> bool {
        *self & Castles::from(Square::A1.perspective(side)) != Castles::none()
    }

    /// The kingside castling square, if side has the rights.
    #[inline(always)]
    pub fn short(&self, side: Color) -> Option<Square> {
        if self.has_short(side) {
            Some(Square::G1.perspective(side))
        } else {
            None
        }
    }

    /// The queenside castling square, if side has the rights.
    #[inline(always)]
    pub fn long(&self, side: Color) -> Option<Square> {
        if self.has_long(side) {
            Some(Square::C1.perspective(side))
        } else {
            None
        }
    }
}

impl Default for Castles {
    #[inline(always)]
    fn default() -> Self {
        Castles::all()
    }
}

impl From<Square> for Castles {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        pub static CASTLES: SyncUnsafeCell<[Castles; 64]> =
            unsafe { MaybeUninit::zeroed().assume_init() };

        #[cold]
        #[ctor::ctor]
        #[inline(never)]
        unsafe fn init() {
            let castles = unsafe { CASTLES.get().as_mut_unchecked() };
            castles[Square::A1 as usize] = Castles(Bits::new(0b0010));
            castles[Square::H1 as usize] = Castles(Bits::new(0b0001));
            castles[Square::E1 as usize] = Castles(Bits::new(0b0011));
            castles[Square::A8 as usize] = Castles(Bits::new(0b1000));
            castles[Square::H8 as usize] = Castles(Bits::new(0b0100));
            castles[Square::E8 as usize] = Castles(Bits::new(0b1100));
        }

        unsafe { CASTLES.get().as_ref_unchecked()[sq as usize] }
    }
}

impl Display for Castles {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for side in Color::iter() {
            if self.has_short(side) {
                Display::fmt(&Piece::new(Role::King, side), f)?;
            }

            if self.has_long(side) {
                Display::fmt(&Piece::new(Role::Queen, side), f)?;
            }
        }

        Ok(())
    }
}

/// The reason why parsing [`Castles`] failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display("failed to parse castling rights")]
pub struct ParseCastlesError;

impl FromStr for Castles {
    type Err = ParseCastlesError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut castles = Castles::none();

        use Piece::*;
        for c in s.chars() {
            let mut buffer = [0; 4];

            match Piece::from_str(c.encode_utf8(&mut buffer)) {
                Ok(p @ WhiteKing | p @ BlackKing) if !castles.has_short(p.color()) => {
                    castles |= Castles::from(Square::H1.perspective(p.color()));
                }

                Ok(p @ WhiteQueen | p @ BlackQueen) if !castles.has_long(p.color()) => {
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
    fn short_returns_kingside_castle_square(cr: Castles, #[filter(#cr.has_short(#c))] c: Color) {
        assert_eq!(cr.short(c), Some(Square::G1.perspective(c)));
    }

    #[proptest]
    fn long_returns_queenside_castle_square(cr: Castles, #[filter(#cr.has_long(#c))] c: Color) {
        assert_eq!(cr.long(c), Some(Square::C1.perspective(c)));
    }

    #[proptest]
    fn parsing_printed_castles_is_an_identity(cr: Castles) {
        assert_eq!(cr.to_string().parse(), Ok(cr));
    }

    #[proptest]
    fn parsing_castles_fails_if_right_is_duplicated(
        #[filter(!#s.is_empty())]
        #[strategy("(KK)?(kk)?(QQ)?(qq)?")]
        s: String,
    ) {
        assert_eq!(Castles::from_str(&s), Err(ParseCastlesError));
    }

    #[proptest]
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

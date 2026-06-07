use crate::util::{Assume, Binary, Bits, Int, Num};
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::ops::{Index, IndexMut};
use std::{hint::unreachable_unchecked, str::FromStr};

/// The type of a chess [`Piece`][`crate::chess::Piece`].
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(u8)]
pub enum Role {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

const unsafe impl Num for Role {
    type Repr = u8;
    const MIN: Self::Repr = Role::Pawn as u8;
    const MAX: Self::Repr = Role::King as u8;
}

const unsafe impl Int for Role {}

const impl Binary for Role {
    type Bits = Bits<u8, 3>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        match self {
            Role::Pawn => Bits::new(0b010),
            Role::Knight => Bits::new(0b011),
            Role::Bishop => Bits::new(0b101),
            Role::Rook => Bits::new(0b110),
            Role::Queen => Bits::new(0b111),
            Role::King => Bits::new(0b001),
        }
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        match bits.get() {
            0b010 => Role::Pawn,
            0b011 => Role::Knight,
            0b101 => Role::Bishop,
            0b110 => Role::Rook,
            0b111 => Role::Queen,
            0b001 => Role::King,
            _ => unsafe { unreachable_unchecked() },
        }
    }
}

impl Display for Role {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Role::Pawn => f.write_char('p'),
            Role::Knight => f.write_char('n'),
            Role::Bishop => f.write_char('b'),
            Role::Rook => f.write_char('r'),
            Role::Queen => f.write_char('q'),
            Role::King => f.write_char('k'),
        }
    }
}

/// The reason why parsing the piece.
#[derive(Debug, Display, Copy, Error)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[display("failed to parse piece")]
pub struct ParseRoleError;

const impl FromStr for Role {
    type Err = ParseRoleError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "p" => Ok(Role::Pawn),
            "n" => Ok(Role::Knight),
            "b" => Ok(Role::Bishop),
            "r" => Ok(Role::Rook),
            "q" => Ok(Role::Queen),
            "k" => Ok(Role::King),
            _ => Err(ParseRoleError),
        }
    }
}

const impl<T> Index<Role> for [T; Role::MAX as usize + 1] {
    type Output = T;

    #[inline(always)]
    fn index(&self, p: Role) -> &Self::Output {
        self.get(p.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Role> for [T; Role::MAX as usize + 1] {
    #[inline(always)]
    fn index_mut(&mut self, p: Role) -> &mut Self::Output {
        self.get_mut(p.cast::<usize>()).assume()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn role_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Role>>(), size_of::<Role>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn decoding_encoded_role_is_an_identity(r: Role) {
        assert_eq!(Role::decode(r.encode()), r);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_role_is_an_identity(r: Role) {
        assert_eq!(r.to_string().parse(), Ok(r));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_role_fails_if_not_one_of_lowercase_pnbrqk(
        #[filter(!['p', 'n', 'b', 'r', 'q', 'k'].contains(&#c))] c: char,
    ) {
        assert_eq!(c.to_string().parse::<Role>(), Err(ParseRoleError));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_role_fails_if_length_not_one(#[filter(#s.len() != 1)] s: String) {
        assert_eq!(s.parse::<Role>(), Err(ParseRoleError));
    }
}

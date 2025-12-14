use crate::util::Int;
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::str::FromStr;

/// The type of a chess [`Piece`][`crate::chess::Piece`].
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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

unsafe impl const Int for Role {
    type Repr = u8;
    const MIN: Self::Repr = Role::Pawn as _;
    const MAX: Self::Repr = Role::King as _;
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
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display("failed to parse piece")]
pub struct ParseRoleError;

impl FromStr for Role {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use test_strategy::proptest;

    #[test]
    fn role_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Role>>(), size_of::<Role>());
    }

    #[proptest]
    fn parsing_printed_role_is_an_identity(r: Role) {
        assert_eq!(r.to_string().parse(), Ok(r));
    }

    #[proptest]
    fn parsing_role_fails_if_not_one_of_lowercase_pnbrqk(
        #[filter(!['p', 'n', 'b', 'r', 'q', 'k'].contains(&#c))] c: char,
    ) {
        assert_eq!(c.to_string().parse::<Role>(), Err(ParseRoleError));
    }

    #[proptest]
    fn parsing_role_fails_if_length_not_one(#[filter(#s.len() != 1)] s: String) {
        assert_eq!(s.parse::<Role>(), Err(ParseRoleError));
    }
}

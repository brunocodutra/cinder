use crate::simd::*;
use crate::util::{Assume, Binary, Bits, Int, Niched, Num};
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::ops::{Index, IndexMut};
use std::str::FromStr;

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

const unsafe impl Niched for Role {}

const impl Role {
    #[expect(dead_code)]
    const REQUIRES: () = const { assert!(size_of::<Self>() == size_of::<Option<Self>>()) };

    pub const LEN: usize = Self::MAX as usize + 1;

    pub const ENCODER: u8x64 = const {
        let mut encoder = [0xFFu8; 16];
        encoder[Role::Pawn as usize] = 0b0010;
        encoder[Role::Knight as usize] = 0b0011;
        encoder[Role::Bishop as usize] = 0b0101;
        encoder[Role::Rook as usize] = 0b0110;
        encoder[Role::Queen as usize] = 0b0111;
        encoder[Role::King as usize] = 0b0001;
        Aligned([encoder; 4]).cast()
    };

    pub const DECODER: u8x64 = const {
        let mut decoder = [0xFFu8; 16];
        decoder[Role::ENCODER.as_array()[Role::Pawn as usize] as usize] = Role::Pawn.get();
        decoder[Role::ENCODER.as_array()[Role::Knight as usize] as usize] = Role::Knight.get();
        decoder[Role::ENCODER.as_array()[Role::Bishop as usize] as usize] = Role::Bishop.get();
        decoder[Role::ENCODER.as_array()[Role::Rook as usize] as usize] = Role::Rook.get();
        decoder[Role::ENCODER.as_array()[Role::Queen as usize] as usize] = Role::Queen.get();
        decoder[Role::ENCODER.as_array()[Role::King as usize] as usize] = Role::King.get();
        Aligned([decoder; 4]).cast()
    };

    /// This role's representation in ASCII.
    #[inline(always)]
    pub fn to_ascii(self) -> u8 {
        b"pnbrqk"[self]
    }
}

const impl Binary for Role {
    type Bits = Bits<u8, 3>;

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

const impl<T> Index<Role> for [T; Role::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, p: Role) -> &Self::Output {
        self.get(p.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Role> for [T; Role::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, p: Role) -> &mut Self::Output {
        self.get_mut(p.cast::<usize>()).assume()
    }
}

impl Display for Role {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char(self.to_ascii() as char)
    }
}

/// The reason why parsing [`Role`] failed.
#[derive(Debug, Display, Copy, Error)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[display("failed to parse role")]
pub struct ParseRoleError;

impl FromStr for Role {
    type Err = ParseRoleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let &[b] = s.as_bytes() else {
            return Err(ParseRoleError);
        };

        Self::iter()
            .find(|r| r.to_ascii() == b)
            .ok_or(ParseRoleError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

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

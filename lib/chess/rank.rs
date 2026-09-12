use crate::chess::{Bitboard, File, Flip, Transpose};
use crate::util::{Assume, Int, Niched, Num};
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::ops::{Index, IndexMut};
use std::{ops::Sub, str::FromStr};

/// A row on the chess board.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(i8)]
pub enum Rank {
    First,
    Second,
    Third,
    Fourth,
    Fifth,
    Sixth,
    Seventh,
    Eighth,
}

const unsafe impl Num for Rank {
    type Repr = i8;
    const MIN: Self::Repr = Rank::First as i8;
    const MAX: Self::Repr = Rank::Eighth as i8;
}

const unsafe impl Int for Rank {}

const unsafe impl Niched for Rank {}

const impl Rank {
    #[expect(dead_code)]
    const REQUIRES: () = const { assert!(size_of::<Self>() == size_of::<Option<Self>>()) };

    pub const LEN: usize = Self::MAX as usize + 1;

    /// Returns a [`Bitboard`] that only contains this rank.
    #[inline(always)]
    pub fn bitboard(self) -> Bitboard {
        Bitboard::new(0x000000000000FF << (self.get() * 8))
    }
}

const impl Flip for Rank {
    /// This rank from the opponent's perspective.
    #[inline(always)]
    fn flip(self) -> Self {
        Self::new(self.get() ^ Self::MAX)
    }
}

const impl Transpose for Rank {
    type Transposition = File;

    /// This rank's corresponding file.
    #[inline(always)]
    fn transpose(self) -> Self::Transposition {
        self.convert().assume()
    }
}

const impl Sub for Rank {
    type Output = i8;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self.get() - rhs.get()
    }
}

const impl<T> Index<Rank> for [T; Rank::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, r: Rank) -> &Self::Output {
        self.get(r.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Rank> for [T; Rank::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, r: Rank) -> &mut Self::Output {
        self.get_mut(r.cast::<usize>()).assume()
    }
}

impl Display for Rank {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char((b'1' + self.cast::<u8>()).into())
    }
}

/// The reason why parsing [`Rank`] failed.
#[derive(Debug, Display, Copy, Error)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[display("failed to parse rank")]
pub struct ParseRankError;

impl FromStr for Rank {
    type Err = ParseRankError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let [b] = s.as_bytes() else {
            return Err(ParseRankError);
        };

        b.checked_sub(b'1')
            .and_then(Num::convert)
            .ok_or(ParseRankError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::Square;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn subtracting_ranks_returns_distance(a: Rank, b: Rank) {
        assert_eq!(a - b, a.get() - b.get());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn flipping_rank_returns_its_complement(r: Rank) {
        assert_eq!(r.flip().get(), Rank::MAX - r.get());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn transposing_rank_returns_its_corresponding_file(r: Rank) {
        assert_eq!(r.transpose().get(), r.get());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn rank_has_an_equivalent_bitboard(r: Rank) {
        assert_eq!(
            Vec::from_iter(r.bitboard()),
            Vec::from_iter(File::iter().map(|f| Square::new(f, r)))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_rank_is_an_identity(r: Rank) {
        assert_eq!(r.to_string().parse(), Ok(r));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_rank_fails_if_not_digit_between_1_and_8(
        #[filter(!('1'..='8').contains(&#c))] c: char,
    ) {
        assert_eq!(c.to_string().parse::<Rank>(), Err(ParseRankError));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_rank_fails_if_length_not_one(#[filter(#s.len() != 1)] s: String) {
        assert_eq!(s.parse::<Rank>(), Err(ParseRankError));
    }
}

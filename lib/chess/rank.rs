use crate::chess::{Bitboard, File, Flip, Transpose};
use crate::util::{Assume, Int};
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::{ops::Sub, str::FromStr};

/// A row on the chess board.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
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

impl Rank {
    /// Returns a [`Bitboard`] that only contains this rank.
    #[inline(always)]
    pub fn bitboard(self) -> Bitboard {
        Bitboard::new(0x000000000000FF << (self.get() * 8))
    }
}

unsafe impl const Int for Rank {
    type Repr = i8;
    const MIN: Self::Repr = Rank::First as _;
    const MAX: Self::Repr = Rank::Eighth as _;
}

impl const Flip for Rank {
    /// This rank from the opponent's perspective.
    #[inline(always)]
    fn flip(self) -> Self {
        Self::new(self.get() ^ Self::MAX)
    }
}

impl const Transpose for Rank {
    type Transposition = File;

    /// This rank's corresponding file.
    #[inline(always)]
    fn transpose(self) -> Self::Transposition {
        self.convert().assume()
    }
}

impl Sub for Rank {
    type Output = i8;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self.get() - rhs.get()
    }
}

impl Display for Rank {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char((b'1' + self.cast::<u8>()).into())
    }
}

/// The reason why parsing [`Rank`] failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display("failed to parse rank")]
pub struct ParseRankError;

impl FromStr for Rank {
    type Err = ParseRankError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let [c] = s.as_bytes() else {
            return Err(ParseRankError);
        };

        c.checked_sub(b'1')
            .and_then(Int::convert)
            .ok_or(ParseRankError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{File, Square};
    use std::mem::size_of;
    use test_strategy::proptest;

    #[test]
    fn rank_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Rank>>(), size_of::<Rank>());
    }

    #[proptest]
    fn subtracting_ranks_returns_distance(a: Rank, b: Rank) {
        assert_eq!(a - b, a.get() - b.get());
    }

    #[proptest]
    fn flipping_rank_returns_its_complement(r: Rank) {
        assert_eq!(r.flip().get(), Rank::MAX - r.get());
    }

    #[proptest]
    fn transposing_rank_returns_its_corresponding_file(r: Rank) {
        assert_eq!(r.transpose().get(), r.get());
    }

    #[proptest]
    fn rank_has_an_equivalent_bitboard(r: Rank) {
        assert_eq!(
            Vec::from_iter(r.bitboard()),
            Vec::from_iter(File::iter().map(|f| Square::new(f, r)))
        );
    }

    #[proptest]
    fn parsing_printed_rank_is_an_identity(r: Rank) {
        assert_eq!(r.to_string().parse(), Ok(r));
    }

    #[proptest]
    fn parsing_rank_fails_if_not_digit_between_1_and_8(
        #[filter(!('1'..='8').contains(&#c))] c: char,
    ) {
        assert_eq!(c.to_string().parse::<Rank>(), Err(ParseRankError));
    }

    #[proptest]
    fn parsing_rank_fails_if_length_not_one(#[filter(#s.len() != 1)] s: String) {
        assert_eq!(s.parse::<Rank>(), Err(ParseRankError));
    }
}

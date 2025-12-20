use crate::chess::{Bitboard, Mirror, Rank, Transpose};
use crate::util::{Assume, Int};
use derive_more::with_trait::{Display, Error};
use std::fmt::{self, Formatter, Write};
use std::{ops::Sub, str::FromStr};

/// A column on the chess board.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq, Ord, PartialOrd)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(i8)]
pub enum File {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
}

impl File {
    /// Returns a [`Bitboard`] that only contains this file.
    #[inline(always)]
    pub const fn bitboard(self) -> Bitboard {
        Bitboard::new(0x0101010101010101 << self.get())
    }
}

unsafe impl const Int for File {
    type Repr = i8;
    const MIN: Self::Repr = File::A as _;
    const MAX: Self::Repr = File::H as _;
}

impl const Mirror for File {
    /// Horizontally mirrors this file.
    #[inline(always)]
    fn mirror(self) -> Self {
        Self::new(self.get() ^ Self::MAX)
    }
}

impl const Transpose for File {
    type Transposition = Rank;

    /// This file's corresponding rank.
    #[inline(always)]
    fn transpose(self) -> Self::Transposition {
        self.convert().assume()
    }
}

impl const Sub for File {
    type Output = i8;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        self.get() - rhs.get()
    }
}

impl Display for File {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char((b'a' + self.cast::<u8>()).into())
    }
}

/// The reason why parsing [`File`] failed.
#[derive(Debug, Display, Error)]
#[derive_const(Default, Clone, Eq, PartialEq)]
#[display("failed to parse file")]
pub struct ParseFileError;

impl FromStr for File {
    type Err = ParseFileError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let [c] = s.as_bytes() else {
            return Err(ParseFileError);
        };

        c.checked_sub(b'a')
            .and_then(Int::convert)
            .ok_or(ParseFileError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::{Rank, Square};
    use std::mem::size_of;
    use test_strategy::proptest;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn file_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<File>>(), size_of::<File>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn mirroring_file_returns_its_complement(f: File) {
        assert_eq!(f.mirror().get(), File::MAX - f.get());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn transposing_file_returns_its_corresponding_rank(f: File) {
        assert_eq!(f.transpose().get(), f.get());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn subtracting_files_returns_distance(a: File, b: File) {
        assert_eq!(a - b, a.get() - b.get());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn file_has_an_equivalent_bitboard(f: File) {
        assert_eq!(
            Vec::from_iter(f.bitboard()),
            Vec::from_iter(Rank::iter().map(|r| Square::new(f, r)))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_file_is_an_identity(f: File) {
        assert_eq!(f.to_string().parse(), Ok(f));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_file_fails_if_not_lower_case_letter_between_a_and_h(
        #[filter(!('a'..='h').contains(&#c))] c: char,
    ) {
        assert_eq!(c.to_string().parse::<File>(), Err(ParseFileError));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_file_fails_if_length_not_one(#[filter(#s.len() != 1)] s: String) {
        assert_eq!(s.parse::<File>(), Err(ParseFileError));
    }
}

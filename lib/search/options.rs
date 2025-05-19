use crate::util::Integer;
use derive_more::with_trait::{Debug, Display, Error, Shl, Shr};
use std::{cmp::Ordering, collections::HashSet, path::PathBuf, str::FromStr};

/// The hash size in bytes.
#[derive(Debug, Display, Copy, Clone, Eq, Ord, Hash, Shl, Shr)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("HashSize({_0})")]
#[display("{}", self.get() >> 20)]
#[repr(transparent)]
pub struct HashSize(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] usize);

unsafe impl Integer for HashSize {
    type Repr = usize;

    const MIN: Self::Repr = 0;

    #[cfg(not(test))]
    const MAX: usize = 1 << 45;

    #[cfg(test)]
    const MAX: usize = 16 << 20;
}

impl Default for HashSize {
    fn default() -> Self {
        HashSize(16 << 20)
    }
}

impl<I: Integer<Repr = usize>> PartialEq<I> for HashSize {
    fn eq(&self, other: &I) -> bool {
        self.get().eq(&other.get())
    }
}

impl<I: Integer<Repr = usize>> PartialOrd<I> for HashSize {
    fn partial_cmp(&self, other: &I) -> Option<Ordering> {
        self.get().partial_cmp(&other.get())
    }
}

/// The reason why parsing the hash size failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display(
    "failed to parse hash size, expected integer in the range `{}..={}`",
    HashSize::lower(),
    HashSize::upper()
)]
pub struct ParseHashSizeError;

impl FromStr for HashSize {
    type Err = ParseHashSizeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<usize>()
            .ok()
            .and_then(|h| h.checked_shl(20))
            .and_then(usize::convert)
            .ok_or(ParseHashSizeError)
    }
}

/// The thread count.
#[derive(Debug, Display, Copy, Clone, Eq, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("ThreadCount({_0})")]
#[display("{_0}")]
#[repr(transparent)]
pub struct ThreadCount(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] u16);

unsafe impl Integer for ThreadCount {
    type Repr = u16;

    const MIN: Self::Repr = 1;

    #[cfg(not(test))]
    const MAX: Self::Repr = 1 << 12;

    #[cfg(test)]
    const MAX: Self::Repr = 4;
}

impl Default for ThreadCount {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<I: Integer<Repr = u16>> PartialEq<I> for ThreadCount {
    fn eq(&self, other: &I) -> bool {
        self.get().eq(&other.get())
    }
}

impl<I: Integer<Repr = u16>> PartialOrd<I> for ThreadCount {
    fn partial_cmp(&self, other: &I) -> Option<Ordering> {
        self.get().partial_cmp(&other.get())
    }
}

/// The reason why parsing the thread count failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display(
    "failed to parse thread count, expected integer in the range `{}..={}`",
    ThreadCount::lower(),
    ThreadCount::upper()
)]
pub struct ParseThreadCountError;

impl FromStr for ThreadCount {
    type Err = ParseThreadCountError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<usize>()
            .ok()
            .and_then(Integer::convert)
            .ok_or(ParseThreadCountError)
    }
}

/// Configuration for adversarial search algorithms.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Options {
    /// The size of the transposition table in bytes.
    ///
    /// This is an upper limit, the actual memory allocation may be smaller.
    pub hash: HashSize,

    /// The number of threads to use while searching.
    pub threads: ThreadCount,

    /// The paths where Syzygy tablebase files are located.
    pub syzygy: HashSet<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn hash_size_is_smaller_than_max(h: HashSize) {
        assert!(HashSize::default() <= HashSize::MAX);
        assert!(h <= HashSize::MAX);
    }

    #[proptest]
    fn hash_size_constructs_if_size_not_too_large(
        #[strategy(HashSize::MIN..=HashSize::MAX)] n: usize,
    ) {
        assert_eq!(HashSize::new(n), n);
    }

    #[proptest]
    fn parsing_printed_hash_size_rounds_to_megabytes(h: HashSize) {
        assert_eq!(h.to_string().parse(), Ok(h >> 20 << 20));
    }

    #[proptest]
    fn parsing_hash_size_fails_for_numbers_too_large(#[strategy(HashSize::MAX + 1..)] n: usize) {
        assert_eq!(n.to_string().parse::<HashSize>(), Err(ParseHashSizeError));
    }

    #[proptest]
    fn parsing_hash_size_fails_for_invalid_number(
        #[filter(#s.parse::<usize>().is_err())] s: String,
    ) {
        assert_eq!(s.to_string().parse::<HashSize>(), Err(ParseHashSizeError));
    }

    #[proptest]
    fn thread_count_is_smaller_than_max(t: ThreadCount) {
        assert!(ThreadCount::default() <= ThreadCount::MAX);
        assert!(t <= ThreadCount::MAX);
    }

    #[proptest]
    fn thread_count_constructs_if_count_not_too_large(
        #[strategy(ThreadCount::MIN..=ThreadCount::MAX)] n: u16,
    ) {
        assert_eq!(ThreadCount::new(n), n);
    }

    #[proptest]
    fn parsing_printed_thread_count_is_an_identity(t: ThreadCount) {
        assert_eq!(t.to_string().parse(), Ok(t));
    }

    #[proptest]
    fn parsing_thread_count_fails_for_numbers_too_large(
        #[strategy(ThreadCount::MAX + 1..)] n: u16,
    ) {
        assert_eq!(
            n.to_string().parse::<ThreadCount>(),
            Err(ParseThreadCountError)
        );
    }

    #[proptest]
    fn parsing_thread_count_fails_for_invalid_number(
        #[filter(#s.parse::<usize>().is_err())] s: String,
    ) {
        assert_eq!(
            s.to_string().parse::<ThreadCount>(),
            Err(ParseThreadCountError)
        );
    }
}

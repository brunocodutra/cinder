use crate::util::{Int, Num};
use derive_more::with_trait::{Debug, Deref, Display, Error, IntoIterator};
use std::ops::{Shl, Shr};
use std::time::Duration;
use std::{cmp::Ordering, collections::HashSet, str::FromStr};

#[cfg(test)]
use proptest::collection::hash_set;

/// The hash size in bytes.
#[derive(Debug, Display, Clone, Copy, Eq, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("HashSize({_0:?})")]
#[display("{}", self.get() >> 20)]
#[repr(transparent)]
pub struct HashSize(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <HashSize as Num>::Repr);

const unsafe impl Num for HashSize {
    type Repr = usize;

    const MIN: Self::Repr = 0;

    #[cfg(not(test))]
    const MAX: usize = 1 << 45;

    #[cfg(test)]
    const MAX: usize = 16 << 20;
}

const unsafe impl Int for HashSize {}

impl HashSize {
    pub const NAME: &str = "Hash";
}

impl Default for HashSize {
    fn default() -> Self {
        HashSize(16 << 20)
    }
}

impl<I: Int<Repr = usize>> PartialEq<I> for HashSize {
    fn eq(&self, other: &I) -> bool {
        self.get().eq(&other.get())
    }
}

impl<I: Int<Repr = usize>> PartialOrd<I> for HashSize {
    fn partial_cmp(&self, other: &I) -> Option<Ordering> {
        self.get().partial_cmp(&other.get())
    }
}

impl Shl<u32> for HashSize {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u32) -> Self::Output {
        Self(self.0.shl(rhs))
    }
}

impl Shr<u32> for HashSize {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0.shr(rhs))
    }
}

/// The reason why parsing the hash size failed.
#[derive(Debug, Display, Default, Clone, Copy, PartialEq, Eq, Error)]
#[display(
    "failed to parse {}, expected integer in the range `{}..={}`",
    HashSize::NAME,
    HashSize::lower(),
    HashSize::upper()
)]
pub struct ParseHashSizeError;

impl FromStr for HashSize {
    type Err = ParseHashSizeError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<<Self as Num>::Repr>()
            .ok()
            .and_then(|h| h.checked_shl(20))
            .and_then(Num::convert)
            .ok_or(ParseHashSizeError)
    }
}

/// The thread count.
#[derive(Debug, Display, Clone, Copy, Eq, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("ThreadCount({_0:?})")]
#[display("{_0}")]
#[repr(transparent)]
pub struct ThreadCount(
    #[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <ThreadCount as Num>::Repr,
);

const unsafe impl Num for ThreadCount {
    type Repr = u16;

    const MIN: Self::Repr = 1;

    #[cfg(not(test))]
    const MAX: Self::Repr = 1 << 12;

    #[cfg(test)]
    const MAX: Self::Repr = 4;
}

const unsafe impl Int for ThreadCount {}

impl ThreadCount {
    pub const NAME: &str = "Threads";
}

impl Default for ThreadCount {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<I: Int<Repr = u16>> PartialEq<I> for ThreadCount {
    fn eq(&self, other: &I) -> bool {
        self.get().eq(&other.get())
    }
}

impl<I: Int<Repr = u16>> PartialOrd<I> for ThreadCount {
    fn partial_cmp(&self, other: &I) -> Option<Ordering> {
        self.get().partial_cmp(&other.get())
    }
}

/// The reason why parsing the thread count failed.
#[derive(Debug, Display, Default, Clone, Copy, PartialEq, Eq, Error)]
#[display(
    "failed to parse {}, expected integer in the range `{}..={}`",
    ThreadCount::NAME,
    ThreadCount::lower(),
    ThreadCount::upper()
)]
pub struct ParseThreadCountError;

impl FromStr for ThreadCount {
    type Err = ParseThreadCountError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<<Self as Num>::Repr>()
            .ok()
            .and_then(Num::convert)
            .ok_or(ParseThreadCountError)
    }
}

/// The move overhead.
#[derive(Debug, Display, Clone, Copy, Eq, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("MoveOverhead({_0:?})")]
#[display("{_0}")]
#[repr(transparent)]
pub struct MoveOverhead(
    #[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <MoveOverhead as Num>::Repr,
);

const unsafe impl Num for MoveOverhead {
    type Repr = u16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 5000;
}

const unsafe impl Int for MoveOverhead {}

impl MoveOverhead {
    pub const NAME: &str = "MoveOverhead";
}

impl Default for MoveOverhead {
    fn default() -> Self {
        Self::new(10)
    }
}

impl<I: Int<Repr = u16>> PartialEq<I> for MoveOverhead {
    fn eq(&self, other: &I) -> bool {
        self.get().eq(&other.get())
    }
}

impl<I: Int<Repr = u16>> PartialOrd<I> for MoveOverhead {
    fn partial_cmp(&self, other: &I) -> Option<Ordering> {
        self.get().partial_cmp(&other.get())
    }
}

impl From<MoveOverhead> for Duration {
    fn from(overhead: MoveOverhead) -> Self {
        Duration::from_millis(overhead.cast())
    }
}

/// The reason why parsing the move overhead failed.
#[derive(Debug, Display, Default, Clone, Copy, PartialEq, Eq, Error)]
#[display(
    "failed to parse {}, expected integer in the range `{}..={}`",
    MoveOverhead::NAME,
    MoveOverhead::lower(),
    MoveOverhead::upper()
)]
pub struct ParseMoveOverheadError;

impl FromStr for MoveOverhead {
    type Err = ParseMoveOverheadError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<<Self as Num>::Repr>()
            .ok()
            .and_then(Num::convert)
            .ok_or(ParseMoveOverheadError)
    }
}

/// The path to Syzygy tablebases.
#[derive(Debug, Display, Default, Clone, PartialEq, Eq, Deref, IntoIterator)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("SyzygyPath({_0:?})")]
#[display("{}", Vec::from_iter(_0.iter().map(String::as_str)).join(Self::PATH_DELIMITER))]
pub struct SyzygyPath(
    #[into_iterator(ref, owned)]
    #[cfg_attr(test, strategy(hash_set("[^:;[:space:]]", 0..5)))]
    HashSet<String>,
);

impl SyzygyPath {
    pub const NAME: &str = "SyzygyPath";

    #[cfg(unix)]
    const PATH_DELIMITER: &str = ":";

    #[cfg(windows)]
    const PATH_DELIMITER: &str = ";";
}

/// The reason why parsing the move overhead failed.
#[derive(Debug, Display, Default, Clone, Copy, PartialEq, Eq, Error)]
#[display(
    "failed to parse {}, expected list of paths separated by `{}`",
    SyzygyPath::NAME,
    SyzygyPath::PATH_DELIMITER
)]
pub struct ParseSyzygyPathError;

impl FromStr for SyzygyPath {
    type Err = ParseSyzygyPathError;

    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(SyzygyPath(HashSet::from_iter(
            s.split(Self::PATH_DELIMITER).filter_map(|s| {
                let trimmed = s.trim_ascii();
                if !trimmed.is_empty() {
                    Some(trimmed.to_owned())
                } else {
                    None
                }
            }),
        )))
    }
}

/// Configuration for adversarial search algorithms.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Options {
    /// The size of the transposition table in bytes.
    ///
    /// This is an upper limit, the actual memory allocation may be smaller.
    pub hash: HashSize,

    /// The number of threads to use while searching.
    pub threads: ThreadCount,

    /// The time assumed to be lost to system latency per move.
    pub overhead: MoveOverhead,

    /// The paths where Syzygy tablebase files are located.
    pub syzygy: SyzygyPath,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn hash_size_constructs_if_not_too_large(#[strategy(HashSize::MIN..=HashSize::MAX)] n: usize) {
        assert_eq!(HashSize::new(n), n);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_hash_size_rounds_to_megabytes(h: HashSize) {
        assert_eq!(h.to_string().parse(), Ok(h >> 20 << 20));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_hash_size_fails_for_numbers_too_large(#[strategy(HashSize::MAX + 1..)] n: usize) {
        assert_eq!(n.to_string().parse::<HashSize>(), Err(ParseHashSizeError));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_hash_size_fails_for_invalid_number(
        #[filter(#s.parse::<usize>().is_err())] s: String,
    ) {
        assert_eq!(s.parse::<HashSize>(), Err(ParseHashSizeError));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn thread_count_constructs_if_not_too_large(
        #[strategy(ThreadCount::MIN..=ThreadCount::MAX)] n: u16,
    ) {
        assert_eq!(ThreadCount::new(n), n);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_thread_count_is_an_identity(t: ThreadCount) {
        assert_eq!(t.to_string().parse(), Ok(t));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_thread_count_fails_for_numbers_too_large(
        #[strategy(ThreadCount::MAX + 1..)] n: u16,
    ) {
        assert_eq!(
            n.to_string().parse::<ThreadCount>(),
            Err(ParseThreadCountError)
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_thread_count_fails_for_invalid_number(
        #[filter(#s.parse::<usize>().is_err())] s: String,
    ) {
        assert_eq!(s.parse::<ThreadCount>(), Err(ParseThreadCountError));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn move_overhead_constructs_if_not_too_large(
        #[strategy(MoveOverhead::MIN..=MoveOverhead::MAX)] n: u16,
    ) {
        assert_eq!(MoveOverhead::new(n), n);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_move_overhead_is_an_identity(o: MoveOverhead) {
        assert_eq!(o.to_string().parse(), Ok(o));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_move_overhead_fails_for_numbers_too_large(
        #[strategy(MoveOverhead::MAX + 1..)] n: u16,
    ) {
        assert_eq!(
            n.to_string().parse::<MoveOverhead>(),
            Err(ParseMoveOverheadError)
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_move_overhead_fails_for_invalid_number(
        #[filter(#s.parse::<usize>().is_err())] s: String,
    ) {
        assert_eq!(s.parse::<MoveOverhead>(), Err(ParseMoveOverheadError));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_printed_syzygy_path_is_an_identity(sp: SyzygyPath) {
        assert_eq!(sp.to_string().parse(), Ok(sp));
    }
}

use crate::chess::{Move, Zobrist};
use crate::search::{Depth, HashSize, Line, Ply, Pv, Score};
use crate::util::{Assume, Binary, Bits, Integer};
use derive_more::with_trait::Debug;
use std::ops::{Index, Range, RangeInclusive};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::{hint::unreachable_unchecked, mem::size_of};

#[cfg(test)]
use crate::chess::Position;

#[cfg(test)]
use proptest::{collection::*, prelude::*};

/// Whether the transposed score is exact or a bound.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum ScoreBound {
    Lower(Score),
    Upper(Score),
    Exact(Score),
}

impl ScoreBound {
    // Constructs a [`ScoreBound`] normalized to [`Ply`].
    #[track_caller]
    #[inline(always)]
    pub fn new(bounds: Range<Score>, score: Score, ply: Ply) -> Self {
        (ply >= 0).assume();
        (bounds.start < bounds.end).assume();

        if score >= bounds.end {
            ScoreBound::Lower(score.normalize(-ply))
        } else if score <= bounds.start {
            ScoreBound::Upper(score.normalize(-ply))
        } else {
            ScoreBound::Exact(score.normalize(-ply))
        }
    }

    // The score bound.
    #[track_caller]
    #[inline(always)]
    pub fn bound(&self, ply: Ply) -> Score {
        (ply >= 0).assume();

        match *self {
            ScoreBound::Lower(s) | ScoreBound::Upper(s) | ScoreBound::Exact(s) => s.normalize(ply),
        }
    }

    /// A lower bound for the score normalized to [`Ply`].
    #[track_caller]
    #[inline(always)]
    pub fn lower(&self, ply: Ply) -> Score {
        (ply >= 0).assume();

        match *self {
            ScoreBound::Upper(_) => Score::mated(ply),
            _ => self.bound(ply),
        }
    }

    /// An upper bound for the score normalized to [`Ply`].
    #[track_caller]
    #[inline(always)]
    pub fn upper(&self, ply: Ply) -> Score {
        (ply >= 0).assume();

        match *self {
            ScoreBound::Lower(_) => Score::mating(ply),
            _ => self.bound(ply),
        }
    }

    /// The score range normalized to [`Ply`].
    #[inline(always)]
    pub fn range(&self, ply: Ply) -> RangeInclusive<Score> {
        self.lower(ply)..=self.upper(ply)
    }
}

impl Binary for ScoreBound {
    type Bits = Bits<u16, { 2 + <Score as Binary>::Bits::BITS }>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        let mut bits = Bits::default();

        match self {
            ScoreBound::Lower(_) => bits.push(Bits::<u8, 2>::new(0b01)),
            ScoreBound::Upper(_) => bits.push(Bits::<u8, 2>::new(0b10)),
            ScoreBound::Exact(_) => bits.push(Bits::<u8, 2>::new(0b11)),
        }

        bits.push(self.bound(Ply::new(0)).encode());

        bits
    }

    #[inline(always)]
    fn decode(mut bits: Self::Bits) -> Self {
        let score = Binary::decode(bits.pop());

        match bits.get() {
            0b01 => ScoreBound::Lower(score),
            0b10 => ScoreBound::Upper(score),
            0b11 => ScoreBound::Exact(score),
            _ => unsafe { unreachable_unchecked() },
        }
    }
}

/// A partial search result.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Transposition {
    score: ScoreBound,
    draft: Depth,
    best: Move,
}

impl Transposition {
    const BITS: u32 = <ScoreBound as Binary>::Bits::BITS
        + <Depth as Binary>::Bits::BITS
        + <Move as Binary>::Bits::BITS;

    /// Constructs a [`Transposition`] given a [`ScoreBound`], the [`Depth`] searched, and the best [`Move`].
    #[inline(always)]
    pub fn new(score: ScoreBound, draft: Depth, best: Move) -> Self {
        Transposition { score, draft, best }
    }

    /// The score bound.
    #[inline(always)]
    pub fn score(&self) -> ScoreBound {
        self.score
    }

    /// The depth searched.
    #[inline(always)]
    pub fn draft(&self) -> Depth {
        self.draft
    }

    /// The principal variation normalized to [`Ply`].
    #[inline(always)]
    pub fn transpose(&self, ply: Ply) -> Pv<1> {
        Pv::new(self.score().bound(ply), Line::singular(self.best))
    }
}

impl Binary for Transposition {
    type Bits = Bits<u64, { Self::BITS }>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        let mut bits = Bits::default();
        bits.push(self.score.encode());
        bits.push(self.draft.encode());
        bits.push(self.best.encode());
        bits
    }

    #[inline(always)]
    fn decode(mut bits: Self::Bits) -> Self {
        Transposition {
            best: Binary::decode(bits.pop()),
            draft: Binary::decode(bits.pop()),
            score: Binary::decode(bits.pop()),
        }
    }
}

type Signature = Bits<u32, { 64 - <Transposition as Binary>::Bits::BITS }>;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
struct SignedTransposition(Signature, <Transposition as Binary>::Bits);

impl Binary for SignedTransposition {
    type Bits = Bits<u64, 64>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        let mut bits = Bits::default();
        bits.push(self.1);
        bits.push(self.0);
        bits
    }

    #[inline(always)]
    fn decode(mut bits: Self::Bits) -> Self {
        SignedTransposition(bits.pop(), bits.pop())
    }
}

/// A cache for [`Transposition`]s.
#[derive(Debug)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("TranspositionTable({})", self.capacity())]
pub struct TranspositionTable {
    #[cfg_attr(test,
        strategy(hash_map(any::<Position>(), any::<Transposition>(), ..32).prop_map(|ts| {
            let mut cache: Box<[AtomicU64]> = (0..ts.len().next_power_of_two()).map(|_| AtomicU64::default()).collect();

            for (pos, t) in ts {
                let key = pos.zobrist();
                let idx = key.slice(..cache.len().trailing_zeros()).cast::<usize>();
                let sig = key.slice(cache.len().trailing_zeros()..).pop();
                *cache[idx].get_mut() = Some(SignedTransposition(sig, t.encode())).encode().get();
            }

            cache
        }))
    )]
    cache: Box<[AtomicU64]>,
}

impl TranspositionTable {
    const WIDTH: usize = size_of::<<Option<SignedTransposition> as Binary>::Bits>();

    /// Constructs a transposition table of at most `size` many bytes.
    #[inline(always)]
    pub fn new(size: HashSize) -> Self {
        let capacity = (1 + size.get() / 2).next_power_of_two() / Self::WIDTH;

        TranspositionTable {
            cache: (0..capacity).map(|_| AtomicU64::default()).collect(),
        }
    }

    /// The actual size of this table in bytes.
    #[inline(always)]
    pub fn size(&self) -> HashSize {
        HashSize::new(self.capacity() * Self::WIDTH)
    }

    /// The actual size of this table in number of entries.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.cache.len()
    }

    /// Instructs the CPU to load the slot associated with `key` onto the cache.
    #[inline(always)]
    pub fn prefetch(&self, key: Zobrist) {
        if self.capacity() > 0 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{_MM_HINT_ET0, _mm_prefetch};
                _mm_prefetch(self.cache[key].as_ptr() as _, _MM_HINT_ET0);
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::{_PREFETCH_LOCALITY0, _PREFETCH_WRITE, _prefetch};
                let ptr = self.cache[key].as_ptr() as _;
                _prefetch(ptr, _PREFETCH_WRITE, _PREFETCH_LOCALITY0);
            }
        }
    }

    /// Loads the [`Transposition`] from the slot associated with `key`.
    #[inline(always)]
    pub fn get(&self, key: Zobrist) -> Option<Transposition> {
        if self.capacity() == 0 {
            return None;
        }

        let sig = self.sign(key);
        let bits = Bits::new(self.cache[key].load(Relaxed));
        match Binary::decode(bits) {
            Some(SignedTransposition(s, t)) if s == sig => Some(Binary::decode(t)),
            _ => None,
        }
    }

    /// Stores a [`Transposition`] in the slot associated with `key`.
    #[inline(always)]
    pub fn set(&self, key: Zobrist, tpos: Transposition) {
        if self.capacity() > 0 {
            let sig = self.sign(key);
            let bits = Some(SignedTransposition(sig, tpos.encode())).encode();
            self.cache[key].store(bits.get(), Relaxed);
        }
    }

    /// Returns the [`Signature`] associated with `key`.
    #[inline(always)]
    pub fn sign(&self, key: Zobrist) -> Signature {
        key.slice(self.capacity().trailing_zeros()..).pop()
    }
}

impl Index<Zobrist> for [AtomicU64] {
    type Output = AtomicU64;

    #[inline(always)]
    fn index(&self, key: Zobrist) -> &Self::Output {
        let idx: usize = key.slice(..self.len().trailing_zeros()).cast();
        self.get(idx).assume()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn bound_returns_score_bound(
        #[filter(!#b.is_empty())] b: Range<Score>,
        s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs()) as _).contains(&#p.get()))] p: Ply,
    ) {
        assert_eq!(ScoreBound::new(b, s, p).bound(p), s);
    }

    #[proptest]
    fn lower_returns_score_lower_bound(
        #[filter(!#b.is_empty())] b: Range<Score>,
        #[filter(#s > #b.start)] s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs()) as _).contains(&#p.get()))] p: Ply,
    ) {
        assert_eq!(ScoreBound::new(b, s, p).lower(p), s);
    }

    #[proptest]
    fn upper_returns_score_upper_bound(
        #[filter(!#b.is_empty())] b: Range<Score>,
        #[filter(#s < #b.end)] s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs()) as _).contains(&#p.get()))] p: Ply,
    ) {
        assert_eq!(ScoreBound::new(b, s, p).upper(p), s);
    }

    #[proptest]
    fn bound_is_within_range(
        #[filter(!#b.is_empty())] b: Range<Score>,
        s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs()) as _).contains(&#p.get()))] p: Ply,
    ) {
        assert!(ScoreBound::new(b, s, p).range(p).contains(&s));
    }

    #[proptest]
    fn decoding_encoded_score_bound_is_an_identity(s: ScoreBound) {
        assert_eq!(ScoreBound::decode(s.encode()), s);
    }

    #[proptest]
    fn transposed_score_is_within_bounds(t: Transposition, #[filter(#p >= 0)] p: Ply) {
        assert!(t.score().range(p).contains(&t.transpose(p).score()));
    }

    #[proptest]
    fn decoding_encoded_transposition_is_an_identity(t: Transposition) {
        assert_eq!(Transposition::decode(t.encode()), t);
    }

    #[proptest]
    fn decoding_encoded_signed_transposition_is_an_identity(t: SignedTransposition) {
        assert_eq!(SignedTransposition::decode(t.encode()), t);
    }

    #[proptest]
    fn table_input_size_is_an_upper_limit(s: HashSize) {
        assert!(TranspositionTable::new(s).size() <= s);
    }

    #[proptest]
    fn table_size_is_exact_if_input_is_power_of_two(
        #[strategy(TranspositionTable::WIDTH.trailing_zeros()..=HashSize::MAX.trailing_zeros())]
        bits: u32,
    ) {
        let s = HashSize::new(1 << bits);
        assert_eq!(TranspositionTable::new(s).size(), s);
    }

    #[proptest]
    fn table_capacity_equals_the_size_in_bytes(tt: TranspositionTable) {
        assert_eq!(tt.size(), tt.cache.len() * TranspositionTable::WIDTH);
    }

    #[proptest]
    fn get_does_nothing_if_capacity_is_zero(k: Zobrist) {
        assert_eq!(TranspositionTable::new(HashSize::new(0)).get(k), None);
    }

    #[proptest]
    fn get_returns_none_if_transposition_does_not_exist(tt: TranspositionTable, k: Zobrist) {
        tt.cache[k].store(0, Relaxed);
        assert_eq!(tt.get(k), None);
    }

    #[proptest]
    fn get_returns_none_if_signature_does_not_match(
        tt: TranspositionTable,
        t: Transposition,
        k: Zobrist,
    ) {
        let st = Some(SignedTransposition(!tt.sign(k), t.encode()));
        tt.cache[k].store(st.encode().get(), Relaxed);
        assert_eq!(tt.get(k), None);
    }

    #[proptest]
    fn get_returns_some_if_transposition_exists(
        tt: TranspositionTable,
        t: Transposition,
        k: Zobrist,
    ) {
        let st = Some(SignedTransposition(tt.sign(k), t.encode()));
        tt.cache[k].store(st.encode().get(), Relaxed);
        assert_eq!(tt.get(k), Some(t));
    }

    #[proptest]
    fn set_does_nothing_if_capacity_is_zero(k: Zobrist, t: Transposition) {
        TranspositionTable::new(HashSize::new(0)).set(k, t);
    }

    #[proptest]
    fn set_replaces_transposition_if_one_exists(
        #[by_ref] tt: TranspositionTable,
        s: Signature,
        t: Transposition,
        u: Transposition,
        k: Zobrist,
    ) {
        let st = Some(SignedTransposition(s, t.encode()));
        tt.cache[k].store(st.encode().get(), Relaxed);
        tt.set(k, u);
        assert_eq!(tt.get(k), Some(u));
    }

    #[proptest]
    fn set_stores_transposition_if_none_exists(
        tt: TranspositionTable,
        t: Transposition,
        k: Zobrist,
    ) {
        tt.cache[k].store(None::<SignedTransposition>.encode().get(), Relaxed);
        tt.set(k, t);
        assert_eq!(tt.get(k), Some(t));
    }
}

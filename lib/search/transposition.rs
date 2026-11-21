use crate::chess::{Halfmove, Move};
use crate::params::Params;
use crate::search::{Depth, Line, Ply, Pv, Score};
use crate::util::{Binary, Bits, Int, Num};
use bytemuck::{NoUninit, Zeroable, zeroed};
use std::ops::{Range, RangeInclusive};
use std::{cmp::Ordering, hint::unreachable_unchecked};

/// The transposition age.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, NoUninit)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Age(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Age as Num>::Repr);

const unsafe impl Num for Age {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 31;
}

const unsafe impl Int for Age {}

impl Age {
    /// How many generations elapse from `self` to `other`.
    #[inline(always)]
    pub fn distance(self, other: Self) -> Self {
        Age::new(other.get().wrapping_sub(self.get()) & Age::MAX)
    }
}

impl Binary for Age {
    type Bits = Bits<u8, 5>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        Num::new(self.get())
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        Num::new(bits.get())
    }
}

/// How good a [`Transposition`] is.
#[derive(Debug, Default, Clone, Copy, PartialEq, Zeroable, NoUninit)]
#[repr(transparent)]
pub struct Quality(f32);

const unsafe impl Num for Quality {
    type Repr = f32;
    const MIN: Self::Repr = -f32::MAX;
    const MAX: Self::Repr = f32::MAX;
}

/// Semi-order that requires overwhelming superiority.
impl PartialOrd for Quality {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let margin = *Params::tt_quality_margin(0);

        #[expect(clippy::float_cmp)]
        if self.0 == other.0 {
            Some(Ordering::Equal)
        } else if self.0 > other.0 + margin {
            Some(Ordering::Greater)
        } else if self.0 + margin < other.0 {
            Some(Ordering::Less)
        } else {
            None
        }
    }
}

/// How much a transposition is worth to the current search.
#[derive(Debug, Default, Clone, Copy, Zeroable, NoUninit)]
#[repr(transparent)]
pub struct Relevance(f32);

const unsafe impl Num for Relevance {
    type Repr = f32;
    const MIN: Self::Repr = -f32::MAX;
    const MAX: Self::Repr = f32::MAX;
}

impl Eq for Relevance {}

impl PartialEq for Relevance {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Ord for Relevance {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for Relevance {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Whether the transposed score is exact or a bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum ScoreBound {
    Lower(Score),
    Upper(Score),
    Exact(Score),
}

impl ScoreBound {
    // Constructs a [`ScoreBound`] normalized to [`Ply`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(bounds: Range<Score>, score: Score, ply: Ply) -> Self {
        debug_assert!(bounds.start < bounds.end);

        if score >= bounds.end {
            ScoreBound::Lower(score.relative_to_root(ply))
        } else if score <= bounds.start {
            ScoreBound::Upper(score.relative_to_root(ply))
        } else {
            ScoreBound::Exact(score.relative_to_root(ply))
        }
    }

    // Constructs a proven lower [`ScoreBound`] normalized to [`Ply`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn lower_bound(score: Score, ply: Ply) -> Self {
        ScoreBound::Lower(score.relative_to_root(ply))
    }

    // Constructs a proven upper [`ScoreBound`] normalized to [`Ply`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn upper_bound(score: Score, ply: Ply) -> Self {
        ScoreBound::Upper(score.relative_to_root(ply))
    }

    // The score bound.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn bound(self, ply: Ply) -> Score {
        match self {
            ScoreBound::Lower(s) | ScoreBound::Upper(s) | ScoreBound::Exact(s) => {
                s.relative_to_ply(ply)
            }
        }
    }

    /// A lower bound for the score normalized to [`Ply`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn lower(self, ply: Ply) -> Score {
        match self {
            ScoreBound::Upper(_) => Score::mated(ply),
            _ => self.bound(ply),
        }
    }

    /// An upper bound for the score normalized to [`Ply`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn upper(self, ply: Ply) -> Score {
        match self {
            ScoreBound::Lower(_) => Score::mating(ply),
            _ => self.bound(ply),
        }
    }

    /// The score range normalized to [`Ply`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn range(self, ply: Ply) -> RangeInclusive<Score> {
        self.lower(ply)..=self.upper(ply)
    }

    /// Lowers scores that cannot be delivered before the 50-move rule.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn downgrade(self, hm: Halfmove) -> Self {
        match self {
            ScoreBound::Lower(s) => ScoreBound::Lower(s.downgrade(hm)),
            ScoreBound::Upper(s) => ScoreBound::Upper(s.downgrade(hm)),
            ScoreBound::Exact(s) => ScoreBound::Exact(s.downgrade(hm)),
        }
    }
}

impl Binary for ScoreBound {
    type Bits = Bits<u16, { 2 + <Score as Binary>::Bits::BITS }>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn encode(&self) -> Self::Bits {
        let mut bits = Bits::default();

        match self {
            ScoreBound::Lower(_) => bits.push(Bits::<u8, 2>::new(0b01)),
            ScoreBound::Upper(_) => bits.push(Bits::<u8, 2>::new(0b10)),
            ScoreBound::Exact(_) => bits.push(Bits::<u8, 2>::new(0b11)),
        }

        bits.push(self.bound(zeroed()).encode());

        bits
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Transposition {
    pub score: ScoreBound,
    pub depth: Depth,
    pub best: Option<Move>,
    pub age: Age,
    pub was_pv: bool,
}

impl Transposition {
    const BITS: u32 = 1
        + <ScoreBound as Binary>::Bits::BITS
        + <Depth as Binary>::Bits::BITS
        + <Move as Binary>::Bits::BITS
        + <Age as Binary>::Bits::BITS;

    /// Constructs a [`Transposition`] given a [`ScoreBound`], the [`Depth`] searched, and the best [`Move`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(score: ScoreBound, depth: Depth, best: Option<Move>, was_pv: bool) -> Self {
        Transposition {
            score,
            depth,
            best,
            age: zeroed(),
            was_pv,
        }
    }

    /// Lowers scores that cannot be delivered before the 50-move rule.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn downgrade(self, halfmove: Halfmove) -> Self {
        Transposition {
            score: self.score.downgrade(halfmove),
            ..self
        }
    }

    /// The principal variation normalized to [`Ply`].
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn transpose(self, ply: Ply) -> Pv<1> {
        Pv::new(
            self.score.bound(ply),
            self.best.map_or_else(Line::empty, Line::singular),
        )
    }

    /// Whether this entry was written during the given `age`.
    #[inline(always)]
    pub fn is_live(self, age: Age) -> bool {
        self.age == age
    }

    /// How good this entry is.
    #[inline(always)]
    pub fn quality(self) -> Quality {
        let is_exact = matches!(self.score, ScoreBound::Exact(_));

        let mut score = 0.0;
        score = Params::tt_quality(0).mul_add(self.depth.cast(), score);
        score = Params::tt_quality(1).mul_add(self.was_pv.cast(), score);
        score = Params::tt_quality(2).mul_add(is_exact.cast(), score);
        Quality::new(score)
    }

    /// How much this entry is worth to the search at a given `age`.
    #[inline(always)]
    pub fn relevance(self, age: Age) -> Relevance {
        let mut score = 0.0;
        score = Params::tt_relevance(0).mul_add(self.depth.cast(), score);
        score = Params::tt_relevance(1).mul_add(self.age.distance(age).cast(), score);
        Relevance::new(score)
    }
}

impl Binary for Transposition {
    type Bits = Bits<u64, { Self::BITS }>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn encode(&self) -> Self::Bits {
        let mut bits = Bits::default();
        bits.push(self.score.encode());
        bits.push(self.depth.encode());
        bits.push(self.best.encode());
        bits.push(self.age.encode());
        bits.push::<u8, 1>(Bits::new(self.was_pv as u8));
        bits
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn decode(mut bits: Self::Bits) -> Self {
        Transposition {
            was_pv: bits.pop::<u8, 1>() == Bits::new(1),
            age: Binary::decode(bits.pop()),
            best: Binary::decode(bits.pop()),
            depth: Binary::decode(bits.pop()),
            score: Binary::decode(bits.pop()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn bound_returns_score_bound(
        #[filter(!#b.is_empty())] b: Range<Score>,
        s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs())).contains(&#p.get()))] p: Ply,
    ) {
        assert_eq!(ScoreBound::new(b, s, p).bound(p), s);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn lower_returns_score_lower_bound(
        #[filter(!#b.is_empty())] b: Range<Score>,
        #[filter(#s > #b.start)] s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs())).contains(&#p.get()))] p: Ply,
    ) {
        assert_eq!(ScoreBound::new(b, s, p).lower(p), s);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn upper_returns_score_upper_bound(
        #[filter(!#b.is_empty())] b: Range<Score>,
        #[filter(#s < #b.end)] s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs())).contains(&#p.get()))] p: Ply,
    ) {
        assert_eq!(ScoreBound::new(b, s, p).upper(p), s);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn bound_is_within_range(
        #[filter(!#b.is_empty())] b: Range<Score>,
        s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs())).contains(&#p.get()))] p: Ply,
    ) {
        assert!(ScoreBound::new(b, s, p).range(p).contains(&s));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn decoding_encoded_score_bound_is_an_identity(s: ScoreBound) {
        assert_eq!(ScoreBound::decode(s.encode()), s);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn decoding_encoded_optional_score_bound_is_an_identity(s: Option<ScoreBound>) {
        assert_eq!(Option::decode(s.encode()), s);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn transposed_score_is_within_bounds(t: Transposition, p: Ply) {
        assert!(t.score.range(p).contains(&t.transpose(p).score()));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn decoding_encoded_transposition_is_an_identity(t: Transposition) {
        assert_eq!(Transposition::decode(t.encode()), t);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn decoding_encoded_optional_transposition_is_an_identity(t: Option<Transposition>) {
        assert_eq!(Option::decode(t.encode()), t);
    }
}

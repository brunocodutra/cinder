use crate::chess::Move;
use crate::search::{Depth, Line, Ply, Pv, Score};
use crate::util::{Assume, Binary, Bits, Integer};
use derive_more::with_trait::Debug;
use std::hint::unreachable_unchecked;
use std::ops::{Range, RangeInclusive};

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
    fn decoding_encoded_optional_score_bound_is_an_identity(s: Option<ScoreBound>) {
        assert_eq!(Option::decode(s.encode()), s);
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
    fn decoding_encoded_optional_transposition_is_an_identity(t: Option<Transposition>) {
        assert_eq!(Option::decode(t.encode()), t);
    }
}

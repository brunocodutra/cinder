use crate::chess::Move;
use crate::search::{Depth, Line, Ply, Pv, Score};
use crate::util::{Assume, Binary, Bits, Int};
use derive_more::with_trait::Debug;
use std::hint::unreachable_unchecked;
use std::ops::{Range, RangeInclusive};

/// Whether the transposed score is exact or a bound.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq, Ord, PartialOrd)]
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
    pub const fn new(bounds: Range<Score>, score: Score, ply: Ply) -> Self {
        (bounds.start < bounds.end).assume();

        if score >= bounds.end {
            ScoreBound::Lower(score.relative_to_root(ply))
        } else if score <= bounds.start {
            ScoreBound::Upper(score.relative_to_root(ply))
        } else {
            ScoreBound::Exact(score.relative_to_root(ply))
        }
    }

    // The score bound.
    #[inline(always)]
    pub const fn bound(&self, ply: Ply) -> Score {
        match *self {
            ScoreBound::Lower(s) | ScoreBound::Upper(s) | ScoreBound::Exact(s) => {
                s.relative_to_ply(ply)
            }
        }
    }

    /// A lower bound for the score normalized to [`Ply`].
    #[inline(always)]
    pub const fn lower(&self, ply: Ply) -> Score {
        match *self {
            ScoreBound::Upper(_) => Score::mated(ply),
            _ => self.bound(ply),
        }
    }

    /// An upper bound for the score normalized to [`Ply`].
    #[inline(always)]
    pub const fn upper(&self, ply: Ply) -> Score {
        match *self {
            ScoreBound::Lower(_) => Score::mating(ply),
            _ => self.bound(ply),
        }
    }

    /// The score range normalized to [`Ply`].
    #[inline(always)]
    pub const fn range(&self, ply: Ply) -> RangeInclusive<Score> {
        self.lower(ply)..=self.upper(ply)
    }
}

impl const Binary for ScoreBound {
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
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Transposition {
    score: ScoreBound,
    depth: Depth,
    best: Option<Move>,
    was_pv: bool,
}

impl Transposition {
    const BITS: u32 = 1
        + <ScoreBound as Binary>::Bits::BITS
        + <Depth as Binary>::Bits::BITS
        + <Move as Binary>::Bits::BITS;

    /// Constructs a [`Transposition`] given a [`ScoreBound`], the [`Depth`] searched, and the best [`Move`].
    #[inline(always)]
    pub const fn new(score: ScoreBound, depth: Depth, best: Option<Move>, was_pv: bool) -> Self {
        Transposition {
            score,
            depth,
            best,
            was_pv,
        }
    }

    /// The score bound.
    #[inline(always)]
    pub const fn score(&self) -> ScoreBound {
        self.score
    }

    /// The depth searched.
    #[inline(always)]
    pub const fn depth(&self) -> Depth {
        self.depth
    }

    /// Whether this position was ever in the PV.
    #[inline(always)]
    pub const fn was_pv(&self) -> bool {
        self.was_pv
    }

    /// The best move.
    #[inline(always)]
    pub const fn best(&self) -> Option<Move> {
        self.best
    }

    /// The principal variation normalized to [`Ply`].
    #[inline(always)]
    pub const fn transpose(&self, ply: Ply) -> Pv<1> {
        Pv::new(
            self.score.bound(ply),
            self.best.map_or_else(Line::empty, Line::singular),
        )
    }
}

impl const Binary for Transposition {
    type Bits = Bits<u64, { Self::BITS }>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        let mut bits = Bits::default();
        bits.push(self.score.encode());
        bits.push(self.depth.encode());
        bits.push(self.best.encode());
        bits.push::<u8, 1>(Bits::new(self.was_pv as _));
        bits
    }

    #[inline(always)]
    fn decode(mut bits: Self::Bits) -> Self {
        Transposition {
            was_pv: bits.pop::<u8, 1>() == Bits::new(1),
            best: Binary::decode(bits.pop()),
            depth: Binary::decode(bits.pop()),
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
    #[cfg_attr(miri, ignore)]
    fn bound_returns_score_bound(
        #[filter(!#b.is_empty())] b: Range<Score>,
        s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs()) as _).contains(&#p.get()))] p: Ply,
    ) {
        assert_eq!(ScoreBound::new(b, s, p).bound(p), s);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn lower_returns_score_lower_bound(
        #[filter(!#b.is_empty())] b: Range<Score>,
        #[filter(#s > #b.start)] s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs()) as _).contains(&#p.get()))] p: Ply,
    ) {
        assert_eq!(ScoreBound::new(b, s, p).lower(p), s);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn upper_returns_score_upper_bound(
        #[filter(!#b.is_empty())] b: Range<Score>,
        #[filter(#s < #b.end)] s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs()) as _).contains(&#p.get()))] p: Ply,
    ) {
        assert_eq!(ScoreBound::new(b, s, p).upper(p), s);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn bound_is_within_range(
        #[filter(!#b.is_empty())] b: Range<Score>,
        s: Score,
        #[filter((0..=(Score::MAX - #s.get().abs()) as _).contains(&#p.get()))] p: Ply,
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
        assert!(t.score().range(p).contains(&t.transpose(p).score()));
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

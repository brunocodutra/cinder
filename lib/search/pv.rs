use crate::chess::Move;
use crate::search::{Depth, Line, Score};
use crate::util::{Assume, Int};
use derive_more::with_trait::Constructor;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, Neg};

/// The principal variation.
#[derive(Debug, Clone, Constructor)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Pv<const N: usize = { Depth::MAX as _ }> {
    score: Score,
    moves: Line<N>,
}

impl<const N: usize> Pv<N> {
    /// An empty principal variation.
    #[inline(always)]
    pub const fn empty(score: Score) -> Self {
        Self::new(score, Line::empty())
    }

    /// The score from the point of view of the side to move.
    #[inline(always)]
    pub const fn score(&self) -> Score {
        self.score
    }

    /// Constrains the score between `lower` and `upper`.
    #[inline(always)]
    pub const fn clamp(self, lower: Score, upper: Score) -> Pv<N> {
        (lower <= upper).assume();
        Pv::new(self.score.clamp(lower, upper), self.moves)
    }

    /// The sequence of [`Move`]s in this principal variation.
    #[inline(always)]
    pub const fn moves(&self) -> &Line<N> {
        &self.moves
    }

    /// Truncates to a principal variation of a different length.
    #[inline(always)]
    pub const fn truncate<const M: usize>(self) -> Pv<M> {
        Pv::new(self.score, self.moves.truncate())
    }

    /// Transposes to a principal variation to a move.
    #[inline(always)]
    pub const fn transpose(self, head: Move) -> Pv<N> {
        Pv::new(self.score, Line::cons(head, self.moves))
    }
}

impl<const N: usize> const Eq for Pv<N> {}

impl<const N: usize> const PartialEq for Pv<N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl<const N: usize> const Ord for Pv<N> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

impl<const N: usize> const PartialOrd for Pv<N> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, const N: usize> const PartialEq<T> for Pv<N>
where
    Score: [const] PartialEq<T>,
{
    #[inline(always)]
    fn eq(&self, other: &T) -> bool {
        self.score.eq(other)
    }
}

impl<T, const N: usize> const PartialOrd<T> for Pv<N>
where
    Score: [const] PartialOrd<T>,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.score.partial_cmp(other)
    }
}

impl<const N: usize> Hash for Pv<N> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.score.hash(state);
    }
}

impl<const N: usize> const Deref for Pv<N> {
    type Target = Line<N>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.moves
    }
}

impl<const N: usize> const Neg for Pv<N> {
    type Output = Self;

    #[inline(always)]
    fn neg(mut self) -> Self::Output {
        self.score = -self.score;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    #[allow(clippy::nonminimal_bool, clippy::double_comparisons)]
    fn pv_ordering_is_consistent(p: Pv, q: Pv) {
        assert_eq!(p == q, p.partial_cmp(&q) == Some(Ordering::Equal));
        assert_eq!(p < q, p.partial_cmp(&q) == Some(Ordering::Less));
        assert_eq!(p > q, p.partial_cmp(&q) == Some(Ordering::Greater));
        assert_eq!(p <= q, p < q || p == q);
        assert_eq!(p >= q, p > q || p == q);
        assert_eq!(p != q, !(p == q));
    }

    #[proptest]
    fn pv_with_larger_score_is_larger(p: Pv, q: Pv) {
        assert_eq!(p < q, p.score() < q.score());
    }

    #[proptest]
    fn clamping_constrains_score_to_interval(pv: Pv, l: Score, #[filter(#r >= #l)] r: Score) {
        assert_eq!(pv.clone().clamp(l, r).score(), pv.score().clamp(l, r));
    }

    #[proptest]
    fn clamping_preserves_moves(pv: Pv, l: Score, #[filter(#r >= #l)] r: Score) {
        assert_eq!(pv.clone().clamp(l, r).moves(), pv.moves());
    }

    #[proptest]
    fn negation_changes_score(pv: Pv) {
        assert_eq!(pv.clone().neg().score(), -pv.score());
    }

    #[proptest]
    fn negation_preserves_moves(pv: Pv) {
        assert_eq!(pv.clone().moves(), pv.neg().moves());
    }

    #[proptest]
    fn truncate_preserves_score(pv: Pv) {
        assert_eq!(pv.score(), pv.truncate::<0>().score());
    }

    #[proptest]
    fn truncate_discards_moves(pv: Pv) {
        assert_eq!(
            &pv.moves().clone().truncate::<2>(),
            pv.truncate::<2>().moves()
        );
    }

    #[proptest]
    fn transpose_preserves_score(pv: Pv, m: Move) {
        assert_eq!(pv.clone().transpose(m).score(), pv.score());
    }

    #[proptest]
    fn transpose_prepends_move(pv: Pv, m: Move) {
        assert_eq!(pv.clone().transpose(m).head(), Some(m));
    }
}

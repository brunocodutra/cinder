use crate::chess::Move;
use crate::search::{Line, Score};
use derive_more::with_trait::{Constructor, Deref};
use std::cmp::Ordering;
use std::ops::Neg;

/// The [principal variation].
///
/// [principal variation]: https://www.chessprogramming.org/Principal_Variation
#[derive(Debug, Clone, Eq, PartialEq, Hash, Deref, Constructor)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Pv<const N: usize> {
    score: Score,
    #[deref]
    moves: Line<N>,
}

impl<const N: usize> Pv<N> {
    /// An empty principal variation.
    #[inline(always)]
    pub fn empty(score: Score) -> Self {
        Self::new(score, Line::empty())
    }

    /// The score from the point of view of the side to move.
    #[inline(always)]
    pub fn score(&self) -> Score {
        self.score
    }

    /// The sequence of [`Move`]s in this principal variation.
    #[inline(always)]
    pub fn moves(&self) -> &Line<N> {
        &self.moves
    }

    /// Truncates to a principal variation of a different length.
    #[inline(always)]
    pub fn truncate<const M: usize>(self) -> Pv<M> {
        Pv::new(self.score, self.moves.truncate())
    }

    /// Transposes to a principal variation to a move.
    #[inline(always)]
    pub fn transpose(self, head: Move) -> Pv<N> {
        Pv::new(self.score, Line::cons(head, self.moves))
    }
}

impl<const N: usize> Ord for Pv<N> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.cmp(&other.score)
    }
}

impl<const N: usize> PartialOrd for Pv<N> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, const N: usize> PartialEq<T> for Pv<N>
where
    Score: PartialEq<T>,
{
    #[inline(always)]
    fn eq(&self, other: &T) -> bool {
        self.score.eq(other)
    }
}

impl<T, const N: usize> PartialOrd<T> for Pv<N>
where
    Score: PartialOrd<T>,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        self.score.partial_cmp(other)
    }
}

impl<const N: usize> Neg for Pv<N> {
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
    fn pv_with_larger_score_is_larger(p: Pv<3>, #[filter(#p.score() != #q.score())] q: Pv<3>) {
        assert_eq!(p < q, p.score() < q.score());
    }

    #[proptest]
    fn negation_changes_score(pv: Pv<3>) {
        assert_eq!(pv.clone().neg().score(), -pv.score());
    }

    #[proptest]
    fn negation_preserves_moves(pv: Pv<3>) {
        assert_eq!(pv.clone().moves(), pv.neg().moves());
    }

    #[proptest]
    fn truncate_preserves_score(pv: Pv<3>) {
        assert_eq!(pv.score(), pv.truncate::<0>().score());
    }

    #[proptest]
    fn truncate_discards_moves(pv: Pv<3>) {
        assert_eq!(
            &pv.moves().clone().truncate::<2>(),
            pv.truncate::<2>().moves()
        );
    }

    #[proptest]
    fn transpose_preserves_score(pv: Pv<3>, m: Move) {
        assert_eq!(pv.clone().transpose(m).score(), pv.score());
    }

    #[proptest]
    fn transpose_prepends_move(pv: Pv<3>, m: Move) {
        assert_eq!(pv.clone().transpose(m).head(), Some(m));
    }
}

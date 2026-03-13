use crate::chess::Move;
use crate::util::{Assume, Bounded, Capacity, ConstCapacity, Num, StaticSeq, StaticSeqIter};
use std::cmp::Reverse;
use std::slice::{Iter, IterMut};

#[cfg(test)]
use proptest::{collection::vec, prelude::*};

/// A measure for how good a [`Move`] is.
pub type Rating = Bounded<i16>;

/// A [`Move`] paired with its [`Rating`].
pub type RatedMove = (Move, Rating);

/// A collection of [`Move`]s.
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Moves {
    #[cfg_attr(test, strategy(vec(any::<RatedMove>(), 0..=10usize)
        .prop_map(StaticSeq::from_iter)))]
    entries: StaticSeq<RatedMove, 255>,

    // Index of the first unsorted move
    #[cfg_attr(test, strategy(Just(0)))]
    unsorted: <ConstCapacity as Capacity>::Usize,
}

impl Moves {
    /// The number of [`Move`]s in this collection.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.entries.len().cast()
    }

    /// Whether there are no [`Move`]s in this collection.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A iterator over the [`RatedMove`]s in this collection in arbitrary order.
    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, RatedMove> {
        self.into_iter()
    }

    /// A mutable iterator over the [`RatedMove`]s in this collection in arbitrary order.
    #[inline(always)]
    pub fn iter_mut(&mut self) -> IterMut<'_, RatedMove> {
        self.into_iter()
    }

    /// An iterator over the [`Move`]s in this collection sorted by their [`Rating`]s.
    #[inline(always)]
    pub fn sorted(&mut self) -> SortedMovesIter<'_> {
        SortedMovesIter::new(self)
    }

    /// Sorts all [`Move`]s in this collection by highest [`Rating`].
    #[inline(always)]
    pub fn sort(&mut self) {
        self.unsorted = self.entries.len().cast();
        self.entries.sort_by_key(|(_, r)| Reverse(*r));
    }

    /// Re-rates all [`Move`]s in this collection.
    #[inline(always)]
    pub fn rate<F: FnMut(Move) -> Rating>(&mut self, mut f: F) {
        for (m, rating) in self {
            *rating = f(*m);
        }
    }
}

impl FromIterator<Move> for Moves {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = Move>>(iter: I) -> Self {
        iter.into_iter().map(|m| (m, Rating::lower())).collect()
    }
}

impl FromIterator<RatedMove> for Moves {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = RatedMove>>(iter: I) -> Self {
        Moves {
            entries: iter.into_iter().collect(),
            unsorted: 0,
        }
    }
}

impl IntoIterator for Moves {
    type Item = RatedMove;
    type IntoIter = StaticSeqIter<RatedMove, 255>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<'a> IntoIterator for &'a Moves {
    type Item = &'a RatedMove;
    type IntoIter = Iter<'a, RatedMove>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

impl<'a> IntoIterator for &'a mut Moves {
    type Item = &'a mut RatedMove;
    type IntoIter = IterMut<'a, RatedMove>;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn into_iter(self) -> Self::IntoIter {
        self.unsorted = 0;
        self.entries.iter_mut()
    }
}

/// A lazily sorted iterator of [`Move`]s.
#[derive(Debug)]
pub struct SortedMovesIter<'a> {
    moves: &'a mut Moves,
    cursor: <ConstCapacity as Capacity>::Usize,
}

impl<'a> SortedMovesIter<'a> {
    #[inline(always)]
    fn new(moves: &'a mut Moves) -> Self {
        SortedMovesIter { moves, cursor: 0 }
    }
}

impl ExactSizeIterator for SortedMovesIter<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.moves.len() - self.cursor.cast::<usize>()
    }
}

impl Iterator for SortedMovesIter<'_> {
    type Item = RatedMove;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn next(&mut self) -> Option<Self::Item> {
        let mut idx = self.cursor.cast::<usize>();
        let mut next = *self.moves.entries.get(idx)?;

        if idx >= self.moves.unsorted.cast::<usize>() {
            let baseline = self.cursor.cast::<usize>() + 1;
            let unsorted = self.moves.entries.get(baseline..).assume();
            for (i, &entry) in unsorted.iter().enumerate() {
                if entry.1 > next.1 {
                    idx = baseline + i;
                    next = entry;
                }
            }
        }

        if self.cursor.cast::<usize>() < idx {
            unsafe { self.moves.entries.swap_unchecked(self.cursor.cast(), idx) }
        }

        self.cursor += 1;
        self.moves.unsorted = self.moves.unsorted.max(self.cursor);
        Some(next)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use test_strategy::proptest;

    #[proptest]
    fn sorted_is_deterministic(mut ms: Moves) {
        assert_eq!(Vec::from_iter(ms.sorted()), Vec::from_iter(ms.sorted()));
    }

    #[proptest]
    fn sorted_iterates_through_moves_by_highest_rating(
        #[filter(#ms.len() == HashSet::<_>::from_iter(#ms.entries.iter().map(|(_, r)| *r)).len())]
        mut ms: Moves,
    ) {
        let ns: Vec<_> = ms.clone().sorted().collect();
        ms.sort();
        assert_eq!(Vec::from_iter(ms), ns);
    }

    #[proptest]
    fn rate_defines_move_order(mut ms: Moves) {
        let mut rating = Rating::new(0);

        ms.rate(|_| {
            rating += 1;
            rating
        });

        let mut ns = Vec::from_iter(ms.clone());
        ns.reverse();

        assert_eq!(Vec::from_iter(ms.sorted()), ns);
    }
}

use crate::chess::Move;
use crate::util::{Assume, Bounded, Capacity, ConstCapacity, Num, StaticSeq};

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
    entries: StaticSeq<RatedMove, 254>,

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

    /// A iterator over the [`Move`]s in this collection in arbitrary order.
    #[inline(always)]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Move> + DoubleEndedIterator {
        self.entries.iter().map(|(m, _)| *m)
    }

    /// An iterator over the [`Move`]s in this collection sorted by their [`Rating`]s.
    #[inline(always)]
    pub fn sorted(&mut self) -> impl ExactSizeIterator<Item = Move> {
        SortedMovesIter::new(self)
    }

    /// Reorders all [`Move`]s in this collection.
    #[inline(always)]
    pub fn sort<F: FnMut(Move) -> Rating>(&mut self, mut f: F) {
        self.unsorted = 0;
        for (m, rating) in &mut self.entries {
            *rating = f(*m);
        }
    }
}

impl FromIterator<Move> for Moves {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn from_iter<I: IntoIterator<Item = Move>>(iter: I) -> Self {
        Moves {
            entries: iter.into_iter().map(|m| (m, Rating::new(0))).collect(),
            unsorted: 0,
        }
    }
}

/// A lazily sorted iterator of [`Move`]s.
#[derive(Debug)]
struct SortedMovesIter<'a> {
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
    type Item = Move;

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
        Some(next.0)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cmp::Reverse, collections::HashSet};
    use test_strategy::proptest;

    #[proptest]
    fn sorting_same_moves_returns_same_sequence(mut ms: Moves) {
        assert_eq!(Vec::from_iter(ms.sorted()), Vec::from_iter(ms.sorted()));
    }

    #[proptest]
    fn sorted_iterates_through_moves_by_highest_rating(
        #[filter(#ms.len() == HashSet::<_>::from_iter(#ms.entries.iter().map(|(_, r)| *r)).len())]
        mut ms: Moves,
    ) {
        let ns: Vec<_> = ms.clone().sorted().collect();
        ms.entries.sort_unstable_by_key(|(_, r)| Reverse(*r));
        assert_eq!(ns, Vec::from_iter(ms.iter()));
    }

    #[proptest]
    fn sort_changes_move_order(ms: Moves) {
        let mut rating = Rating::new(0);

        let mut ns = ms.clone();
        ns.sort(|_| {
            rating += 1;
            rating
        });

        assert_eq!(Vec::from_iter(ns.sorted()), Vec::from_iter(ms.iter().rev()));
    }
}

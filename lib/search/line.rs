use crate::chess::Move;
use derive_more::with_trait::Debug;
use std::fmt::{self, Display, Formatter, Write};

#[cfg(test)]
use proptest::{collection::vec, prelude::*};

/// A sequence of [`Move`]s.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Line({self})")]
pub struct Line<const N: usize>(
    #[cfg_attr(test, strategy(vec(any::<Move>(), ..=N).prop_map(|ms| {
        let mut moves = [None; N];
        for (m, n) in moves.iter_mut().zip(ms) {
            *m = Some(n);
        }
        moves
    })))]
    [Option<Move>; N],
);

impl<const N: usize> Default for Line<N> {
    #[inline(always)]
    fn default() -> Self {
        Self::empty()
    }
}

impl<const N: usize> Line<N> {
    /// An empty [`Line`].
    #[inline(always)]
    pub fn empty() -> Self {
        Line([None; N])
    }

    /// Constructs a singular [`Line`].
    #[inline(always)]
    pub fn singular(m: Move) -> Self {
        Line::cons(m, Line::empty())
    }

    /// Prepends a [`Move`] to a [`Line`].
    #[inline(always)]
    pub fn cons(head: Move, mut tail: Line<N>) -> Self {
        if N > 0 {
            tail.0.copy_within(..N - 1, 1);
            tail.0[0] = Some(head);
        }

        tail
    }

    /// The first [`Move`]s in this [`Line`].
    #[inline(always)]
    pub fn head(&self) -> Option<Move> {
        if N > 0 { self.0[0] } else { None }
    }

    /// An iterator over the [`Move`]s in this [`Line`].
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = Move> {
        self.0.iter().map_while(|m| *m)
    }

    /// Truncates to a principal variation of a different length.
    #[inline(always)]
    pub fn truncate<const M: usize>(self) -> Line<M> {
        let mut line = Line::empty();
        let len = M.min(N);
        if len > 0 {
            line.0[..len].copy_from_slice(&self.0[..len]);
        }

        line
    }
}

impl<const N: usize> Display for Line<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut moves = self.iter();
        let Some(head) = moves.next() else {
            return Ok(());
        };

        Display::fmt(&head, f)?;

        for m in moves {
            f.write_char(' ')?;
            Display::fmt(&m, f)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn cons_truncates_tail(l: Line<3>, m: Move) {
        let cons = Line::<3>::cons(m, l.clone());
        assert_eq!(cons.0[0], Some(m));
        assert_eq!(cons.0[1..], l.0[..2]);
    }

    #[proptest]
    fn head_returns_first_move(l: Line<3>) {
        assert_eq!(l.head(), l.0[0]);
    }

    #[proptest]
    fn truncate_discards_moves(l: Line<3>) {
        assert_eq!(&l.clone().truncate::<2>().0[..], &l.0[..2]);
    }
}

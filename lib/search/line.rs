use crate::chess::Move;
use derive_more::with_trait::Debug;
use std::fmt::{self, Display, Formatter, Write};
use std::ptr::copy;

#[cfg(test)]
use proptest::{collection::vec, prelude::*};

/// A sequence of [`Move`]s.
#[derive(Debug, Clone, Hash)]
#[derive_const(Eq, PartialEq)]
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

impl<const N: usize> const Default for Line<N> {
    #[inline(always)]
    fn default() -> Self {
        Self::empty()
    }
}

impl<const N: usize> Line<N> {
    /// An empty [`Line`].
    #[inline(always)]
    pub const fn empty() -> Self {
        Line([None; N])
    }

    /// Constructs a singular [`Line`].
    #[inline(always)]
    pub const fn singular(m: Move) -> Self {
        Line::cons(m, Line::empty())
    }

    /// Prepends a [`Move`] to a [`Line`].
    #[inline(always)]
    pub const fn cons(head: Move, mut tail: Line<N>) -> Self {
        if N > 0 {
            unsafe {
                let ptr = tail.0.as_mut_ptr();
                copy(ptr, ptr.add(1), N - 1);
                ptr.write(Some(head));
            }
        }

        tail
    }

    /// The first [`Move`]s in this [`Line`].
    #[inline(always)]
    pub const fn head(&self) -> Option<Move> {
        if N > 0 { self.0[0] } else { None }
    }

    /// Truncates to a principal variation of a different length.
    #[inline(always)]
    pub const fn truncate<const M: usize>(self) -> Line<M> {
        let mut line = Line::empty();
        let len = M.min(N);
        if len > 0 {
            line.0[..len].copy_from_slice(&self.0[..len]);
        }

        line
    }

    /// An iterator over the [`Move`]s in this [`Line`].
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = Move> {
        self.0.iter().map_while(|m| *m)
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

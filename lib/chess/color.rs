use crate::{chess::Flip, util::Int};
use derive_more::with_trait::Display;
use std::ops::Not;

/// The color of a chess [`Piece`][`crate::chess::Piece`].
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(u8)]
pub enum Color {
    #[display("white")]
    White,
    #[display("black")]
    Black,
}

unsafe impl Int for Color {
    type Repr = u8;
    const MIN: Self::Repr = Color::White as _;
    const MAX: Self::Repr = Color::Black as _;
}

impl Flip for Color {
    #[inline(always)]
    fn flip(self) -> Self {
        self.not()
    }
}

impl Not for Color {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl From<bool> for Color {
    #[inline(always)]
    fn from(b: bool) -> Self {
        Int::new(b as _)
    }
}

impl From<Color> for bool {
    #[inline(always)]
    fn from(c: Color) -> Self {
        c == Color::Black
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;
    use test_strategy::proptest;

    #[test]
    fn color_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Color>>(), size_of::<Color>());
    }

    #[proptest]
    fn color_has_an_equivalent_boolean(c: Color) {
        assert_eq!(Color::from(bool::from(c)), c);
    }

    #[proptest]
    fn color_implements_not_operator(c: Color) {
        assert_eq!(!c, c.flip());
    }
}

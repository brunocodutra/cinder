use crate::chess::Flip;
use crate::util::{Int, Num};
use derive_more::with_trait::Display;
use std::ops::Not;

/// The color of a chess [`Piece`][`crate::chess::Piece`].
#[derive(Debug, Display, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq, Ord, PartialOrd)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(u8)]
pub enum Color {
    #[display("white")]
    White,
    #[display("black")]
    Black,
}

unsafe impl const Num for Color {
    type Repr = u8;
    const MIN: Self::Repr = Color::White as u8;
    const MAX: Self::Repr = Color::Black as u8;
}

unsafe impl const Int for Color {}

impl const Flip for Color {
    #[inline(always)]
    fn flip(self) -> Self {
        self.not()
    }
}

impl const Not for Color {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl const From<bool> for Color {
    #[inline(always)]
    fn from(b: bool) -> Self {
        Num::new(b as u8)
    }
}

impl const From<Color> for bool {
    #[inline(always)]
    fn from(c: Color) -> Self {
        c == Color::Black
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn color_guarantees_zero_value_optimization() {
        assert_eq!(size_of::<Option<Color>>(), size_of::<Color>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn color_has_an_equivalent_boolean(c: Color) {
        assert_eq!(Color::from(bool::from(c)), c);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn color_implements_not_operator(c: Color) {
        assert_eq!(!c, c.flip());
    }
}

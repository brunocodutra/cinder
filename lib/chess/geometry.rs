use crate::{chess::Color, util::Int};

/// Trait for types that can be seen from a different perspective.
pub trait Perspective<T>: Sized {
    /// This value from `side`'s perspective.
    fn perspective(self, side: T) -> Self;
}

/// Trait for types that can be seen from the opponent's perspective.
pub trait Flip: Sized {
    /// This value from the opponent's perspective.
    fn flip(self) -> Self;
}

impl<T: Flip> Perspective<Color> for T {
    #[inline(always)]
    fn perspective(self, side: Color) -> Self {
        match side {
            Color::White => self,
            Color::Black => self.flip(),
        }
    }
}

/// One of two horizontal perspectives.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(u8)]
pub enum Side {
    Left,
    Right,
}

unsafe impl Int for Side {
    type Repr = u8;
    const MIN: Self::Repr = Side::Left as _;
    const MAX: Self::Repr = Side::Right as _;
}

impl From<bool> for Side {
    #[inline(always)]
    fn from(b: bool) -> Self {
        Int::new(b as _)
    }
}

impl From<Side> for bool {
    #[inline(always)]
    fn from(s: Side) -> Self {
        s == Side::Right
    }
}

/// Trait for types that can be horizontally mirrored.
pub trait Mirror: Sized {
    /// This value's horizontal mirror.
    fn mirror(self) -> Self;
}

impl<T: Mirror> Perspective<Side> for T {
    #[inline(always)]
    fn perspective(self, side: Side) -> Self {
        match side {
            Side::Left => self,
            Side::Right => self.mirror(),
        }
    }
}

/// Trait for types that can be diagonally transposed.
pub trait Transpose: Sized {
    /// This type's diagonal transposition.
    type Transposition;

    /// This value's diagonal transposition.
    fn transpose(self) -> Self::Transposition;
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[proptest]
    fn side_has_an_equivalent_boolean(s: Side) {
        assert_eq!(Side::from(bool::from(s)), s);
    }
}

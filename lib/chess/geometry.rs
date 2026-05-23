use crate::chess::Color;
use crate::util::{Int, Num};

/// Trait for types that can be seen from a different perspective.
pub const trait Perspective<T>: Sized {
    /// This value from `side`'s perspective.
    fn perspective(self, side: T) -> Self;
}

/// Trait for types that can be seen from the opponent's perspective.
pub const trait Flip: Sized {
    /// This value from the opponent's perspective.
    fn flip(self) -> Self;
}

const impl<T: [const] Flip> Perspective<Color> for T {
    #[inline(always)]
    fn perspective(self, side: Color) -> Self {
        match side {
            Color::White => self,
            Color::Black => self.flip(),
        }
    }
}

/// One of two horizontal perspectives.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(u8)]
pub enum Side {
    Left,
    Right,
}

const unsafe impl Num for Side {
    type Repr = u8;
    const MIN: Self::Repr = Side::Left as u8;
    const MAX: Self::Repr = Side::Right as u8;
}

const unsafe impl Int for Side {}

const impl From<bool> for Side {
    #[inline(always)]
    fn from(b: bool) -> Self {
        Num::new(b as u8)
    }
}

const impl From<Side> for bool {
    #[inline(always)]
    fn from(s: Side) -> Self {
        s == Side::Right
    }
}

/// Trait for types that can be horizontally mirrored.
pub const trait Mirror: Sized {
    /// This value's horizontal mirror.
    fn mirror(self) -> Self;
}

const impl<T: [const] Mirror> Perspective<Side> for T {
    #[inline(always)]
    fn perspective(self, side: Side) -> Self {
        match side {
            Side::Left => self,
            Side::Right => self.mirror(),
        }
    }
}

/// Trait for types that can be diagonally transposed.
pub const trait Transpose: Sized {
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
    #[cfg_attr(miri, ignore)]
    fn side_has_an_equivalent_boolean(s: Side) {
        assert_eq!(Side::from(bool::from(s)), s);
    }
}

use crate::chess::Color;

/// Trait for types that can be seen from the opponent's perspective.
pub trait Perspective: Copy {
    /// Flips the perspective.
    fn flip(&self) -> Self;

    /// Sets the perspective to the side of the given [`Color`].
    #[inline(always)]
    fn perspective(&self, side: Color) -> Self {
        match side {
            Color::White => *self,
            Color::Black => self.flip(),
        }
    }
}

/// Trait for types that can be horizontally mirrored.
pub trait Mirror {
    /// This value's mirror.
    fn mirror(&self) -> Self;
}

/// Trait for types that can be diagonally transposed.
pub trait Transpose {
    type Transposition;

    /// This value's transposition.
    fn transpose(&self) -> Self::Transposition;
}

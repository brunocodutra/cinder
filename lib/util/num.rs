use crate::util::Assume;
use bytemuck::{Pod, Zeroable};
use std::{fmt::Debug, marker::Destruct, mem::transmute_copy};

/// Trait for types that represent numeric types.
///
/// # Safety
///
/// Must only be implemented for types that can be safely transmuted to and from [`Num::Repr`].
pub const unsafe trait Num: 'static + Send + Sync + Copy {
    /// The primitive numeric representation.
    type Repr: [const] NumRepr;

    /// The minimum repr.
    const MIN: Self::Repr;

    /// The maximum repr.
    const MAX: Self::Repr;

    /// The minimum value.
    #[inline(always)]
    fn lower() -> Self {
        Self::new(Self::MIN)
    }

    /// The maximum value.
    #[inline(always)]
    fn upper() -> Self {
        Self::new(Self::MAX)
    }

    /// Casts from [`Num::Repr`].
    #[inline(always)]
    fn new(n: Self::Repr) -> Self {
        const { assert!(size_of::<Self>() == size_of::<Self::Repr>()) }
        const { assert!(align_of::<Self>() == align_of::<Self::Repr>()) }

        (Self::MIN..=Self::MAX).contains(&n).assume();
        unsafe { transmute_copy(&n) }
    }

    /// Casts to [`Num::Repr`].
    #[inline(always)]
    fn get(self) -> Self::Repr {
        let repr = unsafe { transmute_copy(&self) };
        (Self::MIN..=Self::MAX).contains(&repr).assume();
        repr
    }

    /// Restricts `self` to the interval `min..=max`.
    #[inline(always)]
    fn clip(self, min: Self, max: Self) -> Self {
        Self::new(self.get().clip(min.get(), max.get()))
    }

    /// Casts to a primitive numeric type.
    ///
    /// This is equivalent to the operator `as`.
    #[inline(always)]
    fn cast<N: NumRepr>(self) -> N {
        self.get().cast()
    }

    /// Converts to another [`Num`], if not out of range.
    #[inline(always)]
    fn convert<N: [const] Num<Repr: [const] NumRepr>>(self) -> Option<N> {
        self.get().convert()
    }

    /// Converts to another [`Num`] with saturation.
    #[inline(always)]
    fn saturate<N: [const] Num<Repr: [const] NumRepr>>(self) -> N {
        self.get().saturate()
    }
}

/// Marker trait for primitive numeric types.
pub const trait NumRepr:
    [const] Num<Repr = Self>
    + Debug
    + [const] Destruct
    + [const] Default
    + [const] PartialEq
    + [const] PartialOrd
    + Zeroable
    + Pod
{
    const IS_FLOAT: bool;
    const IS_SIGNED: bool;
}

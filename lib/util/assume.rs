use std::{hint::assert_unchecked, ptr::NonNull};

/// A trait for types that can be assumed to be another type.
pub const trait Assume {
    /// The type of the assumed value.
    type Assumed;
    /// Assume `Self` represents a value of `Self::Assumed`.
    fn assume(self) -> Self::Assumed;
}

impl const Assume for bool {
    type Assumed = ();

    #[track_caller]
    #[inline(always)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { assert_unchecked(self) }
    }
}

impl<'a, T> const Assume for &'a NonNull<T> {
    type Assumed = &'a T;

    #[track_caller]
    #[inline(always)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { self.as_ref() }
    }
}

impl<'a, T> const Assume for &'a mut NonNull<T> {
    type Assumed = &'a mut T;

    #[track_caller]
    #[inline(always)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { self.as_mut() }
    }
}

impl<T> const Assume for Option<T> {
    type Assumed = T;

    #[track_caller]
    #[inline(always)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { self.unwrap_unchecked() }
    }
}

impl<T, E> const Assume for Result<T, E> {
    type Assumed = T;

    #[track_caller]
    #[inline(always)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { self.unwrap_unchecked() }
    }
}

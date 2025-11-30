use std::{hint::assert_unchecked, ptr::NonNull};

/// A trait for types that can be assumed to be another type.
pub trait Assume {
    /// The type of the assumed value.
    type Assumed;
    /// Assume `Self` represents a value of `Self::Assumed`.
    fn assume(self) -> Self::Assumed;
}

impl Assume for bool {
    type Assumed = ();

    #[track_caller]
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { assert_unchecked(self) }
    }
}

impl<'a, T> Assume for &'a NonNull<T> {
    type Assumed = &'a T;

    #[track_caller]
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { self.as_ref() }
    }
}

impl<'a, T> Assume for &'a mut NonNull<T> {
    type Assumed = &'a mut T;

    #[track_caller]
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { self.as_mut() }
    }
}

impl<T> Assume for Option<T> {
    type Assumed = T;

    #[track_caller]
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { self.unwrap_unchecked() }
    }
}

impl<T, E> Assume for Result<T, E> {
    type Assumed = T;

    #[track_caller]
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn assume(self) -> Self::Assumed {
        // Definitely not safe, but we'll assume unit tests will catch everything.
        unsafe { self.unwrap_unchecked() }
    }
}

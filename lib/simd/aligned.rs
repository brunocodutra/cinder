use bytemuck::Zeroable;
use std::mem::{transmute, transmute_copy};
use std::ops::{Deref, DerefMut};

#[derive(Debug, Copy, Hash, Zeroable)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(align(64))]
pub struct Aligned<T>(pub T);

const impl<T> Aligned<T> {
    /// Transmutes `&self` to `&U`.
    #[inline(always)]
    pub fn cast_ref<U>(&self) -> &U {
        const { assert!(align_of::<Self>() >= align_of::<U>()) }
        const { assert!(size_of::<T>() == size_of::<U>()) }
        unsafe { transmute::<&T, &U>(&self.0) }
    }

    /// Transmutes `&mut self` to `&mut U`.
    #[inline(always)]
    pub fn cast_mut<U>(&mut self) -> &mut U {
        const { assert!(align_of::<Self>() >= align_of::<U>()) }
        const { assert!(size_of::<T>() == size_of::<U>()) }
        unsafe { transmute::<&mut T, &mut U>(&mut self.0) }
    }

    /// Transmutes `&self` to `U` by copy.
    #[inline(always)]
    pub fn cast<U>(&self) -> U {
        const { assert!(size_of::<T>() == size_of::<U>()) }
        unsafe { transmute_copy::<T, U>(&self.0) }
    }
}

const impl<T> Deref for Aligned<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const impl<T> DerefMut for Aligned<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

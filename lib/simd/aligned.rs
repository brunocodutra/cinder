use bytemuck::Zeroable;
use derive_more::with_trait::{Deref, DerefMut, IntoIterator};
use std::mem::transmute;

#[derive(
    Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Zeroable, Deref, DerefMut, IntoIterator,
)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(align(64))]
pub struct Aligned<T>(#[into_iterator(owned, ref, ref_mut)] pub T);

impl<T> Aligned<T> {
    /// Transmutes `&self` to a `&U`.
    #[track_caller]
    #[inline(always)]
    pub const fn cast<U>(&self) -> &U {
        const { assert!(align_of::<Self>() >= align_of::<U>()) }
        const { assert!(size_of::<T>() == size_of::<U>()) }
        unsafe { transmute::<&T, &U>(&self.0) }
    }

    /// Transmutes `&mut self` to `&mut U`.
    #[track_caller]
    #[inline(always)]
    pub const fn cast_mut<U>(&mut self) -> &mut U {
        const { assert!(align_of::<Self>() >= align_of::<U>()) }
        const { assert!(size_of::<T>() == size_of::<U>()) }
        unsafe { transmute::<&mut T, &mut U>(&mut self.0) }
    }
}

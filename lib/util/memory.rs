use crate::util::{Assume, Int, Unsigned};
use bytemuck::{NoUninit, Zeroable, try_zeroed_slice_box, zeroed};
use derive_more::with_trait::Debug;
use std::ops::{Deref, DerefMut, Mul};
use std::{marker::ConstParamTy, mem::MaybeUninit, process::abort, ptr::NonNull, slice};

/// Trait for types that represent raw memory allocation.
pub trait Memory<T>: Sized + AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]> {
    /// The index type.
    type Usize: Unsigned;

    /// Allocates zeroed memory for at least `capacity` objects of type `T`.
    fn zeroed(capacity: Self::Usize) -> Self;

    /// Allocates possibly uninitialized memory for at least `capacity` objects of type `T`.
    #[inline(always)]
    fn uninit(capacity: Self::Usize) -> Self {
        Self::zeroed(capacity)
    }
}

/// Heap-allocated memory for objects of type `T`.
pub type DynamicMemory<T> = Box<[MaybeUninit<T>]>;

impl<T> Memory<T> for DynamicMemory<T> {
    type Usize = usize;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn zeroed(capacity: Self::Usize) -> Self {
        try_zeroed_slice_box(capacity).unwrap_or_else(|()| abort())
    }
}

/// Stack-allocated memory with capacity for up to `N` objects of type `T`.
#[derive(Debug)]
#[debug("StaticMemory({:?})", self.as_ref())]
#[debug(bounds(T: Debug))]
#[repr(transparent)]
pub struct StaticMemory<T, const N: usize>([MaybeUninit<T>; N]);

impl<T, const N: usize> Memory<T> for StaticMemory<T, N> {
    type Usize = u32;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn zeroed(capacity: Self::Usize) -> Self {
        if capacity.cast::<usize>() <= N {
            StaticMemory(MaybeUninit::zeroed().transpose())
        } else {
            abort()
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn uninit(capacity: Self::Usize) -> Self {
        if capacity.cast::<usize>() <= N {
            Default::default()
        } else {
            abort()
        }
    }
}

impl<T, const N: usize> const Default for StaticMemory<T, N> {
    #[inline(always)]
    fn default() -> Self {
        StaticMemory(MaybeUninit::uninit().transpose())
    }
}

impl<T, const N: usize> const AsRef<[MaybeUninit<T>]> for StaticMemory<T, N> {
    #[inline(always)]
    fn as_ref(&self) -> &[MaybeUninit<T>] {
        self.0.as_slice()
    }
}

impl<T, const N: usize> const AsMut<[MaybeUninit<T>]> for StaticMemory<T, N> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.0.as_mut_slice()
    }
}

/// Const-allocated memory of `S` bytes.
#[derive(Debug, Copy, Hash, ConstParamTy, Zeroable)]
#[derive_const(Clone, Eq, PartialEq)]
#[debug("ConstMemory({_0:#04X?})")]
#[repr(C, align(4))]
pub struct ConstMemory<const S: usize>([u8; S]);

impl<T: NoUninit, const S: usize> Memory<T> for ConstMemory<S> {
    type Usize = u8;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn zeroed(capacity: Self::Usize) -> Self {
        const { assert!(align_of::<T>() <= align_of::<Self>()) }

        if capacity.cast::<usize>() * size_of::<T>() <= S {
            Default::default()
        } else {
            abort()
        }
    }
}

impl<const S: usize> const Default for ConstMemory<S> {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl<T, const S: usize> const AsRef<[MaybeUninit<T>]> for ConstMemory<S> {
    #[inline(always)]
    fn as_ref(&self) -> &[MaybeUninit<T>] {
        const { assert!(size_of::<T>() > 0) }
        unsafe { slice::from_raw_parts(self.0.as_ptr().cast(), S / size_of::<T>()) }
    }
}

impl<T: NoUninit, const S: usize> const AsMut<[MaybeUninit<T>]> for ConstMemory<S> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [MaybeUninit<T>] {
        const { assert!(size_of::<T>() > 0) }
        unsafe { slice::from_raw_parts_mut(self.0.as_mut_ptr().cast(), S / size_of::<T>()) }
    }
}

impl<const S: usize> const Deref for ConstMemory<S> {
    type Target = [u8];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const S: usize> const DerefMut for ConstMemory<S> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: NoUninit, const N: usize, const S: usize> const From<[T; N]> for ConstMemory<S> {
    #[inline(always)]
    fn from(data: [T; N]) -> Self {
        const { assert!(size_of::<[T; N]>() <= size_of::<ConstMemory<S>>()) }
        const { assert!(align_of::<[T; N]>() <= align_of::<ConstMemory<S>>()) }

        let mut mem = ConstMemory([0; S]);
        let size = size_of_val(&data);
        let dst = mem.get_mut(..size).assume();
        let src = unsafe { slice::from_raw_parts(data.as_ptr().cast(), size) };
        dst.copy_from_slice(src);
        mem
    }
}

#[derive(Debug, Copy, Hash, Zeroable)]
#[derive_const(Clone, Eq, PartialEq)]
#[repr(C, align(2097152))]
struct Thp([u8; 2097152]);

/// A huge page aligned memory allocation.
#[derive(Debug)]
pub struct HugePage<T> {
    ptr: *mut MaybeUninit<T>,
    capacity: usize,
    pages: usize,
}

unsafe impl<T: Send> Send for HugePage<T> {}
unsafe impl<T: Sync> Sync for HugePage<T> {}

impl<T> Memory<T> for HugePage<T> {
    type Usize = usize;

    /// Allocates a memory mapped block of memory for `len` instances of `T`.
    ///
    /// Advises the operating system to use huge pages and optimize for random access order where possible.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn zeroed(capacity: Self::Usize) -> Self {
        const { assert!(align_of::<T>() <= align_of::<Thp>()) }

        let pages = size_of::<T>().mul(capacity).div_ceil(size_of::<Thp>());
        let boxed = DynamicMemory::<Thp>::zeroed(pages);

        #[allow(dead_code)]
        let size = size_of_val(boxed.deref());
        let ptr = Box::into_raw(boxed);

        unsafe {
            #[cfg(not(miri))]
            #[cfg(target_os = "linux")]
            libc::madvise(ptr.cast(), size, libc::MADV_HUGEPAGE);
        };

        HugePage {
            ptr: ptr.cast(),
            capacity,
            pages,
        }
    }
}

impl<T> Drop for HugePage<T> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn drop(&mut self) {
        unsafe {
            drop(DynamicMemory::<Thp>::from_non_null(NonNull::from_mut(
                slice::from_raw_parts_mut(self.ptr.cast(), self.pages),
            )));
        }
    }
}

impl<T> AsRef<[MaybeUninit<T>]> for HugePage<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[MaybeUninit<T>] {
        unsafe { slice::from_raw_parts(self.ptr, self.capacity) }
    }
}

impl<T> AsMut<[MaybeUninit<T>]> for HugePage<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.capacity) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn huge_page_is_aligned_to_thp(#[strategy(..100usize)] n: usize) {
        let mem = HugePage::<u64>::uninit(n);

        assert!(mem.as_ref().len() >= n);
        assert!(mem.as_ref().as_ptr().is_aligned_to(align_of::<Thp>()));
    }
}

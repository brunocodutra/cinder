use crate::util::{Assume, Int, Unsigned};
use bytemuck::{NoUninit, Zeroable, zeroed};
use derive_more::with_trait::Debug;
use memmap2::{MmapMut, MmapOptions};
use std::ops::{Deref, DerefMut};
use std::{marker::ConstParamTy, mem::MaybeUninit, process::abort, ptr, slice};

pub trait Prefetch {
    fn prefetch(self);
}

impl<T> Prefetch for *const T {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn prefetch(self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_ET0, _mm_prefetch};
            _mm_prefetch(self.cast(), _MM_HINT_ET0);
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::{_PREFETCH_LOCALITY0, _PREFETCH_WRITE, _prefetch};
            _prefetch(self.cast(), _PREFETCH_WRITE, _PREFETCH_LOCALITY0);
        }
    }
}

/// Trait for types that represent raw memory allocation.
pub trait Memory<T>: Sized + AsRef<[MaybeUninit<T>]> + AsMut<[MaybeUninit<T>]> {
    /// The index type.
    type Usize: Unsigned;

    /// Allocates zeroed memory for at least `capacity` objects of type `T`.
    fn zeroed(capacity: Self::Usize) -> Self;

    /// Re-allocates zeroed memory for at least `capacity` objects of type `T` in place.
    #[inline(always)]
    fn zeroed_in_place(&mut self, capacity: Self::Usize) {
        unsafe { ptr::drop_in_place(self) }; // IMPORTANT: deallocate first
        unsafe { ptr::write(self, Self::zeroed(capacity)) };
    }

    /// Allocates possibly uninitialized memory for at least `capacity` objects of type `T`.
    #[inline(always)]
    fn uninit(capacity: Self::Usize) -> Self {
        Self::zeroed(capacity)
    }

    /// Re-allocates possibly uninitialized memory for at least `capacity` objects of type `T`. in place.
    #[inline(always)]
    fn uninit_in_place(&mut self, capacity: Self::Usize) {
        unsafe { ptr::drop_in_place(self) }; // IMPORTANT: deallocate first
        unsafe { ptr::write(self, Self::uninit(capacity)) };
    }
}

/// Stack-allocated memory with capacity for up to `N` objects of type `T`.
#[derive(Debug, Zeroable)]
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

impl<T: Zeroable, const N: usize> const Deref for StaticMemory<T, N> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { self.as_ref().assume_init_ref() }
    }
}

impl<T: Zeroable, const N: usize> const DerefMut for StaticMemory<T, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.as_mut().assume_init_mut() }
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

/// A huge page aligned memory allocation.
#[derive(Debug)]
pub struct HugePages<T> {
    ptr: *mut MaybeUninit<T>,
    capacity: usize,

    #[expect(dead_code)]
    mmap: MmapMut,
}

/// Transparent huge page size.
const THP: usize = 2 << 20;

unsafe impl<T: Send> Send for HugePages<T> {}
unsafe impl<T: Sync> Sync for HugePages<T> {}

impl<T> Memory<T> for HugePages<T> {
    type Usize = usize;

    /// Allocates an anonymous memory map for `capacity` instances of `T`.
    ///
    /// Advises the operating system where possible to use transparent huge pages.
    #[inline(always)]
    fn zeroed(capacity: Self::Usize) -> Self {
        const { assert!(align_of::<T>() <= THP) }

        let size = (capacity * size_of::<T>() + THP - 1).next_multiple_of(THP);

        let mut mmap = MmapOptions::new()
            .len(size)
            .no_reserve_swap()
            .map_anon()
            .unwrap_or_else(|_| abort());

        #[cfg(target_os = "linux")]
        mmap.advise(memmap2::Advice::HugePage).ok(); // best-effort

        let ptr = mmap.as_mut_ptr();
        let offset = ptr.align_offset(THP);

        HugePages {
            ptr: ptr.wrapping_add(offset).cast(),
            capacity,
            mmap,
        }
    }
}

impl<T> const AsRef<[MaybeUninit<T>]> for HugePages<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[MaybeUninit<T>] {
        unsafe { slice::from_raw_parts(self.ptr, self.capacity) }
    }
}

impl<T> const AsMut<[MaybeUninit<T>]> for HugePages<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.capacity) }
    }
}

impl<T: Zeroable> const Deref for HugePages<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { self.as_ref().assume_init_ref() }
    }
}

impl<T: Zeroable> const DerefMut for HugePages<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.as_mut().assume_init_mut() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn huge_pages_can_be_zero_initialized(#[strategy(..10usize)] n: usize) {
        let mem = HugePages::<u32>::zeroed(n);
        assert!(mem.iter().all(|x| *x == 0));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn huge_pages_can_be_reinitialized_in_place(
        #[strategy(..10usize)] m: usize,
        #[strategy(..10usize)] n: usize,
    ) {
        let mut mem = HugePages::<u32>::zeroed(m);

        assert_eq!(mem.len(), m);
        mem.zeroed_in_place(n);
        assert_eq!(mem.len(), n);

        assert!(mem.iter().all(|x| *x == 0));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn huge_pages_are_aligned_to_thp(#[strategy(..100usize)] n: usize) {
        let mem = HugePages::<u64>::uninit(n);

        assert!(mem.as_ref().len() >= n);
        assert!(mem.as_ref().as_ptr().is_aligned_to(THP));
    }
}

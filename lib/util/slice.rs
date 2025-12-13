use bytemuck::Zeroable;
use memmap2::{MmapMut, MmapOptions};
use std::mem::{MaybeUninit, forget, needs_drop, replace};
use std::ops::{Deref, DerefMut};
use std::{alloc::Layout, io, slice};

const HUGEPAGE_SIZE: usize = 2 << 20;

/// A memory mapped slice of `T`.
///
/// Memory is guaranteed to be zero-initialized on all platforms.
#[derive(Debug)]
pub struct Slice<T> {
    len: usize,
    ptr: *const T,
    mmap: MaybeUninit<MmapMut>,
}

unsafe impl<T: Send> Send for Slice<T> {}
unsafe impl<T: Sync> Sync for Slice<T> {}

impl<T: Zeroable> Slice<T> {
    /// Allocates a memory mapped block of memory for `len` instances of `T`.
    ///
    /// Advises the operating system to use huge pages and optimize for random access order where possible.
    pub fn new(len: usize) -> io::Result<Self> {
        const { assert!(size_of::<T>() > 0) }
        const { assert!(!needs_drop::<T>()) }

        let layout = Layout::array::<T>(len).map_err(|_| io::ErrorKind::OutOfMemory)?;
        let alignment = layout.align().max(HUGEPAGE_SIZE);
        let size = (layout.size() + alignment - 1).next_multiple_of(HUGEPAGE_SIZE);
        let mmap = MmapOptions::new().len(size).map_anon()?;

        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Random)?;

        #[cfg(target_os = "linux")]
        mmap.advise(memmap2::Advice::HugePage).ok(); // best-effort

        let ptr = mmap.as_ptr();
        let offset = ptr.align_offset(alignment);

        Ok(Slice {
            len,
            ptr: ptr.wrapping_add(offset) as _,
            mmap: MaybeUninit::new(mmap),
        })
    }

    #[inline(always)]
    pub fn resize(&mut self, len: usize) -> io::Result<()> {
        unsafe { self.mmap.assume_init_drop() }; // IMPORTANT: deallocate before reallocating
        let old = replace(self, Self::new(len)?);
        forget(old); // SAFETY: Drop for Self assumes self.mmap is initialized
        Ok(())
    }
}

impl<T> Drop for Slice<T> {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe { self.mmap.assume_init_drop() };
    }
}

impl<T> Deref for Slice<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for Slice<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.ptr.cast_mut(), self.len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[derive(Zeroable)]
    #[repr(align(4096))]
    struct OverAligned(#[allow(dead_code)] [u8; 4096]);

    #[proptest]
    fn is_aligned_to_type(#[strategy(1..512usize)] n: usize) {
        let slice: Slice<OverAligned> = Slice::new(n)?;

        assert_eq!(slice.len(), n);
        assert!(slice.as_ptr().is_aligned());

        let mmap = unsafe { slice.mmap.assume_init_ref() };
        assert!(mmap.len().is_multiple_of(HUGEPAGE_SIZE));
    }

    #[proptest]
    fn is_aligned_to_page(#[strategy(..HUGEPAGE_SIZE)] n: usize) {
        let slice: Slice<u32> = Slice::new(n)?;

        assert_eq!(slice.len(), n);
        assert!(slice.as_ptr().is_aligned_to(HUGEPAGE_SIZE));

        let mmap = unsafe { slice.mmap.assume_init_ref() };
        assert!(mmap.len().is_multiple_of(HUGEPAGE_SIZE));
    }

    #[proptest]
    fn is_initialized_with_zero(#[strategy(..HUGEPAGE_SIZE)] n: usize) {
        let slice: Slice<u32> = Slice::new(n)?;
        assert!(slice.iter().all(|&x| x == 0));
    }

    #[proptest]
    fn is_mutable(#[strategy(1..HUGEPAGE_SIZE)] n: usize, #[strategy(0..#n)] i: usize, x: u32) {
        let mut slice: Slice<u32> = Slice::new(n)?;

        assert_eq!(slice[i], 0);
        slice[i] = x;
        assert_eq!(slice[i], x);
    }

    #[proptest]
    fn can_be_resized(
        #[strategy(..HUGEPAGE_SIZE)] n: usize,
        #[strategy(..HUGEPAGE_SIZE)] m: usize,
    ) {
        let mut slice: Slice<u32> = Slice::new(n)?;

        assert_eq!(slice.len(), n);
        slice.resize(m)?;
        assert_eq!(slice.len(), m);
    }
}

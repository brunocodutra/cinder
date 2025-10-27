use bytemuck::Zeroable;
use memmap2::{MmapMut, MmapOptions};
use std::ops::{Deref, DerefMut};
use std::{alloc::Layout, io, slice};

const HUGEPAGE_SIZE: usize = 1 << 21;

/// A hugepage-backed slice of `T`.
///
/// Memory is guaranteed to be zero-initialized on all platforms.
#[derive(Debug)]
pub struct Slice<T> {
    ptr: *const T,
    len: usize,
    _mmap: MmapMut,
}

unsafe impl<T: Send> Send for Slice<T> {}
unsafe impl<T: Sync> Sync for Slice<T> {}

impl<T: Zeroable> Slice<T> {
    /// Allocates hugepage-backed block of memory for `len` instances of `T`.
    pub fn new(len: usize) -> io::Result<Self> {
        let layout = Layout::array::<T>(len).map_err(|_| io::ErrorKind::OutOfMemory)?;
        let align = layout.align().max(64); // align to cache line
        let size = (layout.size() + align - 1).next_multiple_of(HUGEPAGE_SIZE);
        let mmap = MmapOptions::new().len(size).populate().map_anon()?;

        #[cfg(target_os = "linux")]
        mmap.advise(memmap2::Advice::HugePage)?;

        let ptr = mmap.as_ptr();
        let offset = ptr.align_offset(align);

        Ok(Slice {
            ptr: ptr.wrapping_add(offset) as _,
            len,
            _mmap: mmap,
        })
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
        unsafe { slice::from_raw_parts_mut(self.ptr as _, self.len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    #[derive(Zeroable)]
    #[repr(align(4096))]
    struct AlignedTo4096(#[allow(dead_code)] [u8; 4096]);

    #[proptest]
    fn is_aligned_to_type(#[strategy(1usize..1024)] n: usize) {
        let slice: Slice<AlignedTo4096> = Slice::new(n)?;

        assert_eq!(slice.len(), n);
        assert!(slice.as_ptr().is_aligned());
        assert!(slice._mmap.len().is_multiple_of(HUGEPAGE_SIZE));
    }

    #[proptest]
    fn is_aligned_to_cache_line(#[strategy(1usize..1024)] n: usize) {
        let slice: Slice<u32> = Slice::new(n)?;

        assert_eq!(slice.len(), n);
        assert!(slice.as_ptr().is_aligned_to(64));
        assert!(slice._mmap.len().is_multiple_of(HUGEPAGE_SIZE));
    }

    #[proptest]
    fn is_initialized_with_zero(#[strategy(1..(1usize << 20))] n: usize) {
        let slice: Slice<u32> = Slice::new(n)?;
        assert!(slice.iter().all(|&x| x == 0));
    }

    #[proptest]
    fn is_mutable(#[strategy(1..(1usize << 20))] n: usize, #[strategy(0..#n)] i: usize, x: u32) {
        let mut slice: Slice<u32> = Slice::new(n)?;

        assert_eq!(slice[i], 0);
        slice[i] = x;
        assert_eq!(slice[i], x);
    }
}

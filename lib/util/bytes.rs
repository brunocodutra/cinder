use crate::util::Aligned;
use bytemuck::{NoUninit, zeroed};
use std::{marker::ConstParamTy, slice};

/// A const statically allocated byte buffer.
#[derive(Debug, Copy, Clone, Eq, PartialEq, ConstParamTy)]
pub struct Bytes<const N: usize> {
    len: u32,
    buffer: Aligned<[u8; N]>,
}

impl<const N: usize> Bytes<N> {
    #[inline(always)]
    pub const fn new<T: NoUninit>(data: &[T]) -> Self {
        let len = size_of_val(data);
        debug_assert!(len <= N);

        let mut buffer: Aligned<[u8; N]> = zeroed();

        let bytes = unsafe { slice::from_raw_parts(data.as_ptr().cast(), len) };
        unsafe { buffer.0.get_unchecked_mut(..len).copy_from_slice(bytes) };

        Bytes {
            len: len as _,
            buffer,
        }
    }

    #[inline(always)]
    pub const fn as_slice<T: NoUninit>(&self) -> &[T] {
        debug_assert!(self.len.is_multiple_of(size_of::<T>() as _));

        unsafe {
            slice::from_raw_parts(
                self.buffer.0.as_ptr().cast(),
                self.len as usize / size_of::<T>(),
            )
        }
    }

    #[inline(always)]
    pub const fn as_mut_slice<T: NoUninit>(&mut self) -> &mut [T] {
        debug_assert!(self.len.is_multiple_of(size_of::<T>() as _));

        unsafe {
            slice::from_raw_parts_mut(
                self.buffer.0.as_mut_ptr().cast(),
                self.len as usize / size_of::<T>(),
            )
        }
    }
}

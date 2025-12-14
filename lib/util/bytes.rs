use bytemuck::NoUninit;
use derive_more::with_trait::Debug;
use std::marker::{ConstParamTy, PhantomData};
use std::mem::needs_drop;
use std::ops::{Deref, DerefMut};
use std::slice;

#[derive(Copy, Hash, ConstParamTy)]
#[derive_const(Clone, Eq, PartialEq)]
#[repr(align(4))]
struct AlignedTo4<T>(T);

/// A const allocated byte buffer.
///
/// This buffer can hold up to `N` bytes in total.
#[derive(Debug, Copy, Hash, ConstParamTy)]
#[derive_const(Clone, Eq, PartialEq)]
#[debug("ByteBuffer({:04X?})", &buffer.0[..*len as usize])]
#[repr(C)]
pub struct ByteBuffer<const N: usize> {
    len: u32,
    buffer: AlignedTo4<[u8; N]>,
}

impl<const N: usize> ByteBuffer<N> {
    /// Stores `data` as raw bytes on the stack, erasing the type `T`.
    #[inline(always)]
    const fn new<T: NoUninit>(data: &[T]) -> Self {
        const { assert!(align_of::<T>() <= align_of::<AlignedTo4<[u8; N]>>()) }
        const { assert!(!needs_drop::<T>()) }

        let len = size_of_val(data);
        debug_assert!(len <= N);

        let mut buffer = AlignedTo4([0; N]);
        let data = unsafe { slice::from_raw_parts(data.as_ptr().cast(), len) };
        unsafe { buffer.0.get_unchecked_mut(..len).copy_from_slice(data) };

        ByteBuffer {
            len: len as _,
            buffer,
        }
    }
}

/// A statically allocated typed byte buffer.
///
/// This buffer can hold up to `N` bytes in total.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[debug("TypedByteBuffer({:?})", self.deref())]
#[debug(bounds(T: Debug))]
pub struct TypedByteBuffer<T: NoUninit, const N: usize> {
    bytes: ByteBuffer<N>,
    phantom: PhantomData<T>,
}

impl<T: NoUninit, const N: usize> TypedByteBuffer<T, N> {
    /// Stores `data` on the stack`.
    #[inline(always)]
    pub const fn new(data: &[T]) -> Self {
        TypedByteBuffer {
            bytes: ByteBuffer::new(data),
            phantom: PhantomData,
        }
    }

    /// Converts to raw bytes, erasing the type `T`.
    #[inline(always)]
    pub const fn into_bytes(self) -> ByteBuffer<N> {
        self.bytes
    }

    /// Converts from raw bytes, reifying the type `T`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee bytes contain objects of type `T`.
    #[inline(always)]
    pub const unsafe fn from_bytes(bytes: ByteBuffer<N>) -> Self {
        TypedByteBuffer {
            bytes,
            phantom: PhantomData,
        }
    }
}

impl<T: NoUninit, const N: usize> const Deref for TypedByteBuffer<T, N> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe {
            slice::from_raw_parts(
                self.bytes.buffer.0.as_ptr().cast(),
                self.bytes.len as usize / size_of::<T>(),
            )
        }
    }
}

impl<T: NoUninit, const N: usize> const DerefMut for TypedByteBuffer<T, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            slice::from_raw_parts_mut(
                self.bytes.buffer.0.as_mut_ptr().cast(),
                self.bytes.len as usize / size_of::<T>(),
            )
        }
    }
}

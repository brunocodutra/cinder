use crate::util::Assume;
use bytemuck::Zeroable;
use std::simd::{prelude::*, *};
use std::{hash::Hash, marker::PhantomData, ops::*};

/// A vector mask for `N` elements.
#[derive(Debug, Default, Clone, Copy)]
#[repr(transparent)]
pub struct M<T: MaskElement, const N: usize> {
    phantom: PhantomData<Mask<T, N>>,
    #[cfg(target_feature = "avx512f")]
    inner: u64,
    #[cfg(not(target_feature = "avx512f"))]
    inner: Mask<T, N>,
}

unsafe impl<T: MaskElement + Zeroable, const N: usize> Zeroable for M<T, N> {}

impl<T: MaskElement + Eq, const N: usize> Eq for M<T, N> {}

impl<T: MaskElement + PartialEq, const N: usize> PartialEq for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: MaskElement + Hash, const N: usize> Hash for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_bitmask().hash(state);
    }
}

impl<T: MaskElement, const N: usize> M<T, N> {
    /// Returns the number of elements enabled in this mask.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn count(self) -> u32 {
        self.to_bitmask().count_ones()
    }

    /// Whether the `i`th element is set.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn test(self, i: usize) -> bool {
        (i < N).assume();

        #[cfg(target_feature = "avx512f")]
        {
            self.inner & 1u64.shl(i as u32) != 0
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            self.inner.test(i)
        }
    }

    /// Whether any element is set.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn any(self) -> bool {
        #[cfg(target_feature = "avx512f")]
        {
            self.inner != 0
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            self.inner.any()
        }
    }

    /// A mask with elements rotated by `n`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn rotate_left<const M: usize>(self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            self.inner.rotate_left(M as u32).into()
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            self.inner.rotate_elements_left::<{ M }>().into()
        }
    }

    /// Rotates the bitboard.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn rotate_right<const M: usize>(self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            self.inner.rotate_right(M as u32).into()
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            self.inner.rotate_elements_right::<{ M }>().into()
        }
    }

    /// Converts to a scalar bitmask.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn to_bitmask(self) -> u64 {
        #[cfg(target_feature = "avx512f")]
        {
            self.inner
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            self.inner.to_bitmask()
        }
    }

    /// Converts from a scalar bitmask.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn from_bitmask(bitmask: u64) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            M {
                phantom: PhantomData,
                inner: bitmask,
            }
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            Mask::from_bitmask(bitmask).into()
        }
    }

    /// Converts to the equivalent [`Simd`] vector.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn to_simd(self) -> Simd<T, N> {
        Mask::from(self).to_simd()
    }

    /// Converts a vector of integers to a mask.
    ///
    /// # Safety
    ///
    /// All elements must be either 0 or -1.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub unsafe fn from_simd_unchecked(simd: Simd<T, N>) -> Self {
        unsafe { Mask::from_simd_unchecked(simd).into() }
    }
}

impl M8x64 {
    /// Floods ranks that are occupied.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn flood_ranks(self) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            let x = self.inner.wrapping_add(0x7E7E7E7E7E7E7E7E) & 0x8080808080808080;
            x.wrapping_sub(x >> 7).into()
        }

        #[cfg(not(target_feature = "avx512f"))]
        unsafe {
            use std::mem::transmute;
            let flooded = transmute::<i8x64, i64x8>(self.to_simd()).simd_ne(Simd::splat(0));
            Self::from_simd_unchecked(transmute::<i64x8, i8x64>(flooded.to_simd()))
        }
    }
}

impl<T: SimdElement, U: MaskElement, const N: usize> Select<Simd<T, N>> for M<U, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn select(self, t: Simd<T, N>, f: Simd<T, N>) -> Simd<T, N> {
        self.inner.select(t, f)
    }
}

impl<T: MaskElement, const N: usize> Not for M<T, N> {
    type Output = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn not(self) -> Self::Output {
        self.inner.not().into()
    }
}

impl<T: MaskElement, const N: usize> BitAnd for M<T, N> {
    type Output = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.inner.bitand(rhs.inner).into()
    }
}

impl<T: MaskElement, const N: usize> BitAnd<Mask<T, N>> for M<T, N> {
    type Output = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitand(self, rhs: Mask<T, N>) -> Self::Output {
        self.bitand(Self::from(rhs))
    }
}

impl<T: MaskElement, const N: usize> BitAndAssign for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.inner.bitand_assign(rhs.inner);
    }
}

impl<T: MaskElement, const N: usize> BitAndAssign<Mask<T, N>> for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitand_assign(&mut self, rhs: Mask<T, N>) {
        self.bitand_assign(Self::from(rhs));
    }
}

impl<T: MaskElement, const N: usize> BitOr for M<T, N> {
    type Output = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.inner.bitor(rhs.inner).into()
    }
}

impl<T: MaskElement, const N: usize> BitOr<Mask<T, N>> for M<T, N> {
    type Output = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitor(self, rhs: Mask<T, N>) -> Self::Output {
        self.bitor(Self::from(rhs))
    }
}

impl<T: MaskElement, const N: usize> BitOrAssign for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.inner.bitor_assign(rhs.inner);
    }
}

impl<T: MaskElement, const N: usize> BitOrAssign<Mask<T, N>> for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitor_assign(&mut self, rhs: Mask<T, N>) {
        self.bitor_assign(Self::from(rhs));
    }
}

impl<T: MaskElement, const N: usize> BitXor for M<T, N> {
    type Output = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.inner.bitxor(rhs.inner).into()
    }
}

impl<T: MaskElement, const N: usize> BitXor<Mask<T, N>> for M<T, N> {
    type Output = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitxor(self, rhs: Mask<T, N>) -> Self::Output {
        self.bitxor(Self::from(rhs))
    }
}

impl<T: MaskElement, const N: usize> BitXorAssign for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.inner.bitxor_assign(rhs.inner);
    }
}

impl<T: MaskElement, const N: usize> BitXorAssign<Mask<T, N>> for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn bitxor_assign(&mut self, rhs: Mask<T, N>) {
        self.bitxor_assign(Self::from(rhs));
    }
}

#[cfg(target_feature = "avx512f")]
impl<T: MaskElement, const N: usize> From<u64> for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn from(bitmask: u64) -> Self {
        Self::from_bitmask(bitmask)
    }
}

impl<T: MaskElement, const N: usize> From<M<T, N>> for u64 {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn from(m: M<T, N>) -> Self {
        m.to_bitmask()
    }
}

impl<T: MaskElement, const N: usize> From<Mask<T, N>> for M<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn from(mask: Mask<T, N>) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            mask.to_bitmask().into()
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            M {
                phantom: PhantomData,
                inner: mask,
            }
        }
    }
}

impl<T: MaskElement, const N: usize> From<M<T, N>> for Mask<T, N> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn from(m: M<T, N>) -> Self {
        #[cfg(target_feature = "avx512f")]
        {
            Mask::from_bitmask(m.inner)
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            m.inner
        }
    }
}

pub type M8x4 = M<i8, 4>;
pub type M8x8 = M<i8, 8>;
pub type M8x16 = M<i8, 16>;
pub type M8x32 = M<i8, 32>;
pub type M8x64 = M<i8, 64>;
pub type M16x4 = M<i16, 4>;
pub type M16x8 = M<i16, 8>;
pub type M16x16 = M<i16, 16>;
pub type M16x32 = M<i16, 32>;
pub type M16x64 = M<i16, 64>;
pub type M32x4 = M<i32, 4>;
pub type M32x8 = M<i32, 8>;
pub type M32x16 = M<i32, 16>;
pub type M32x32 = M<i32, 32>;
pub type M32x64 = M<i32, 64>;
pub type M64x4 = M<i64, 4>;
pub type M64x8 = M<i64, 8>;
pub type M64x16 = M<i64, 16>;
pub type M64x32 = M<i64, 32>;
pub type M64x64 = M<i64, 64>;

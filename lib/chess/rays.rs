use crate::chess::Square;
use crate::simd::*;
use crate::util::{Int, Num};
use derive_more::with_trait::{Debug, Deref};

const RAYS: [Aligned<[u8; 64]>; Square::MAX as usize + 1] = const {
    let mut rays = [Aligned([0x88; 64]); Square::MAX as usize + 1];

    for sq in Square::iter() {
        let ray = &mut rays[sq];

        #[rustfmt::skip]
        let jumps = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];

        #[rustfmt::skip]
        let steps = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];

        let mut i = 8;
        while i > 0 {
            i -= 1;

            let (df, dr) = jumps[i];
            if let Some((file, rank)) = Option::zip(
                (sq.file().get() + df).convert(),
                (sq.rank().get() + dr).convert(),
            ) {
                ray[i * 8] = Square::new(file, rank).cast();
            }

            let mut j = 1;
            let mut sq = sq;
            let (df, dr) = steps[i];
            while let Some((file, rank)) = Option::zip(
                (sq.file().get() + df).convert(),
                (sq.rank().get() + dr).convert(),
            ) {
                sq = Square::new(file, rank);
                ray[i * 8 + j] = sq.cast();
                j += 1;
            }
        }
    }

    rays
};

/// Set of squares reachable by a piece from a source square.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq)]
pub struct Rays(Square);

impl Rays {
    /// Returns the corresponding [`InvRays`].
    #[inline(always)]
    pub fn inv(self) -> InvRays {
        InvRays(self.0)
    }

    /// Returns a bitboard for the places in this ray.
    #[inline(always)]
    pub fn valid(self) -> M8x64 {
        self.simd_ne(Simd::splat(0x88)).into()
    }

    /// Returns a bitboard for the sliding places in this ray.
    #[inline(always)]
    pub fn pins(self) -> M8x64 {
        self.valid() & M8x64::from_bitmask(0xFEFEFEFEFEFEFEFE)
    }
}

const impl From<Square> for Rays {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        Self(sq)
    }
}

const impl Deref for Rays {
    type Target = u8x64;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        RAYS[self.0 as usize].cast_ref()
    }
}

const RAYS_INV: [Aligned<[u8; 64]>; Square::MAX as usize + 1] = const {
    let mut inv_rays = [Aligned([0x88; 64]); Square::MAX as usize + 1];

    for sq in Square::iter() {
        let inv_ray = &mut inv_rays[sq];
        let ray = RAYS[sq];
        for sq in Square::iter() {
            if ray[sq] != 0x88 {
                inv_ray[ray[sq] as usize] = sq as u8;
            }
        }
    }

    inv_rays
};

/// The inverse of [`Rays`].
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq)]
pub struct InvRays(Square);

impl InvRays {
    /// Returns the corresponding [`Rays`].
    #[inline(always)]
    pub fn inv(self) -> Rays {
        Rays(self.0)
    }

    /// Returns the corresponding [`FlippedInvRays`].
    #[inline(always)]
    pub fn flip(self) -> FlippedInvRays {
        FlippedInvRays(self.0)
    }

    /// Returns a bitboard for the elements in this ray.
    #[inline(always)]
    pub fn valid(self) -> M8x64 {
        self.simd_ne(Simd::splat(0x88)).into()
    }
}

const impl From<Square> for InvRays {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        Self(sq)
    }
}

const impl Deref for InvRays {
    type Target = u8x64;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        RAYS_INV[self.0 as usize].cast_ref()
    }
}

const RAYS_INV_FLIPPED: [Aligned<[u8; 64]>; Square::MAX as usize + 1] = const {
    let mut inv_rays = [Aligned([0x88; 64]); Square::MAX as usize + 1];

    for sq in Square::iter() {
        let inv_ray = &mut inv_rays[sq];
        let ray = RAYS[sq];
        for sq in Square::iter() {
            if ray[(sq as usize + 32) % 64] != 0x88 {
                inv_ray[ray[(sq as usize + 32) % 64] as usize] = sq as u8;
            }
        }
    }

    inv_rays
};

/// The flipped inverse of [`Rays`].
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq)]
pub struct FlippedInvRays(Square);

impl FlippedInvRays {
    /// Returns the corresponding [`InvRays`].
    #[inline(always)]
    pub fn flip(self) -> InvRays {
        InvRays(self.0)
    }

    /// Returns a bitboard for the elements in this ray.
    #[inline(always)]
    pub fn valid(self) -> M8x64 {
        self.simd_ne(Simd::splat(0x88)).into()
    }
}

const impl From<Square> for FlippedInvRays {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        Self(sq)
    }
}

const impl Deref for FlippedInvRays {
    type Target = u8x64;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        RAYS_INV_FLIPPED[self.0 as usize].cast_ref()
    }
}

/// Trait for types that can be furled in along [`Rays`].
pub trait Furl: Sized {
    type Furled: Sized;

    /// This value furled in along `rays`.
    fn furl(&self, rays: Rays) -> Self::Furled;
}

impl Furl for u8x64 {
    type Furled = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn furl(&self, rays: Rays) -> Self::Furled {
        self.permute(*rays)
    }
}

impl Furl for M8x64 {
    type Furled = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn furl(&self, rays: Rays) -> Self::Furled {
        let furled = self.to_simd().cast::<u8>().furl(rays).cast::<i8>();
        unsafe { Self::from_simd_unchecked(furled) }
    }
}

/// Trait for types that can be unfurled out along [`Rays`].
pub trait Unfurl: Sized {
    type Unfurled: Sized;

    /// This value unfurled out along `rays`.
    fn unfurl(&self, rays: Rays) -> Self::Unfurled;

    /// This value unfurled out along `rays` and flipped vertically.
    fn unfurl_flip(&self, rays: Rays) -> Self::Unfurled;
}

impl Unfurl for u8x64 {
    type Unfurled = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn unfurl(&self, rays: Rays) -> Self::Unfurled {
        self.permute(*rays.inv())
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn unfurl_flip(&self, rays: Rays) -> Self::Unfurled {
        self.permute(*rays.inv().flip())
    }
}

impl Unfurl for M8x64 {
    type Unfurled = Self;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn unfurl(&self, rays: Rays) -> Self::Unfurled {
        let unfurled = self.to_simd().cast::<u8>().unfurl(rays).cast::<i8>();
        unsafe { Self::from_simd_unchecked(unfurled) }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn unfurl_flip(&self, rays: Rays) -> Self::Unfurled {
        let unfurled = self.to_simd().cast::<u8>().unfurl_flip(rays).cast::<i8>();
        unsafe { Self::from_simd_unchecked(unfurled) }
    }
}

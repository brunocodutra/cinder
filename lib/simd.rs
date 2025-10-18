mod halve;
mod mul4x8;
mod muladd4x8;
mod mulhi;
mod pack;
mod powi;

pub use halve::*;
pub use mul4x8::*;
pub use muladd4x8::*;
pub use mulhi::*;
pub use pack::*;
pub use powi::*;

pub use std::simd::{StdFloat, prelude::*};

#[cfg(target_feature = "avx512f")]
const WIDTH: usize = 64;

#[cfg(not(target_feature = "avx512f"))]
#[cfg(target_feature = "avx2")]
const WIDTH: usize = 32;

#[cfg(not(target_feature = "avx512f"))]
#[cfg(not(target_feature = "avx2"))]
const WIDTH: usize = 16;

pub const W1: usize = WIDTH / 8;
pub const W2: usize = WIDTH / 4;
pub const W4: usize = WIDTH / 2;
pub const W8: usize = WIDTH;

pub type V1<T> = Simd<T, W1>;
pub type V2<T> = Simd<T, W2>;
pub type V4<T> = Simd<T, W4>;
pub type V8<T> = Simd<T, W8>;

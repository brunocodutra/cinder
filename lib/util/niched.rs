use std::num::*;

/// Marker trait for types that contain niches and guarantee zero-value optimization.
///
/// # Safety
///
/// Must only be implemented for types where `size_of::<Self>() == size_of::<Option<Self>>()`.
pub const unsafe trait Niched {}

const unsafe impl Niched for NonZeroU8 {}
const unsafe impl Niched for NonZeroI8 {}
const unsafe impl Niched for NonZeroU16 {}
const unsafe impl Niched for NonZeroI16 {}
const unsafe impl Niched for NonZeroU32 {}
const unsafe impl Niched for NonZeroI32 {}
const unsafe impl Niched for NonZeroU64 {}
const unsafe impl Niched for NonZeroI64 {}
const unsafe impl Niched for NonZeroU128 {}
const unsafe impl Niched for NonZeroI128 {}
const unsafe impl Niched for NonZeroUsize {}
const unsafe impl Niched for NonZeroIsize {}

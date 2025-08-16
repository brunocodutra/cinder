use bytemuck::Zeroable;
use derive_more::with_trait::{Deref, DerefMut, IntoIterator};

#[derive(
    Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Zeroable, Deref, DerefMut, IntoIterator,
)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(align(64))]
pub struct AlignTo64<T>(#[into_iterator(owned, ref, ref_mut)] pub T);

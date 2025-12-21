use bytemuck::{NoUninit, Zeroable};
use derive_more::with_trait::{Debug, Deref, DerefMut};

#[derive(Debug, Default, Deref, DerefMut)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(bound(T)))]
#[debug("{_0:?}")]
#[repr(transparent)]
pub struct Atomic<T: NoUninit>(#[cfg_attr(test, map(atomic::Atomic::new))] atomic::Atomic<T>);

unsafe impl<T: NoUninit + Zeroable> Zeroable for Atomic<T> {}

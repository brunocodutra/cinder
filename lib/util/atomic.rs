use bytemuck::{NoUninit, Zeroable};
use derive_more::{Deref, DerefMut};

#[derive(Debug, Default, Deref, DerefMut)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(bound(T)))]
#[repr(transparent)]
pub struct Atomic<T: NoUninit>(#[cfg_attr(test, map(atomic::Atomic::new))] atomic::Atomic<T>);

unsafe impl<T: NoUninit + Zeroable> Zeroable for Atomic<T> {}

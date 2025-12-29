use crate::util::{Assume, Atomic, Binary, Bits, HugeSeq, Int, Unsigned, zero};
use bytemuck::{Pod, Zeroable};
use derive_more::with_trait::{Debug, Deref, DerefMut};
use std::ops::{Index, IndexMut};
use std::{marker::PhantomData, sync::atomic::Ordering};

#[cfg(test)]
use proptest::{collection::*, prelude::*};

/// The key to a [`Vault`].
pub type Key = Bits<u64, 64>;

impl<T> const Index<Key> for [T] {
    type Output = T;

    #[inline(always)]
    fn index(&self, key: Key) -> &Self::Output {
        let idx = ((key.cast::<u128>() * self.len().cast::<u128>()) >> 64) as usize;
        self.get(idx).assume()
    }
}

impl<T> const IndexMut<Key> for [T] {
    #[inline(always)]
    fn index_mut(&mut self, key: Key) -> &mut Self::Output {
        let idx = ((key.cast::<u128>() * self.len().cast::<u128>()) >> 64) as usize;
        self.get_mut(idx).assume()
    }
}

/// A checksum-guarded container.
#[derive(Debug)]
#[debug("Vault({bits:?})")]
#[repr(transparent)]
pub struct Vault<T: Binary, U: Unsigned = <<T as Binary>::Bits as Int>::Repr> {
    bits: U,
    phantom: PhantomData<T>,
}

unsafe impl<T: Binary, U: Unsigned> Zeroable for Vault<T, U> {}
unsafe impl<T: Binary, U: Unsigned> Pod for Vault<T, U> {}

impl<T: Binary, U: Unsigned> Copy for Vault<T, U> {}

impl<T: Binary, U: Unsigned> Clone for Vault<T, U> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, U, R, const M: u32> Vault<T, U>
where
    T: Binary<Bits = Bits<R, M>>,
    U: Unsigned,
    R: Unsigned,
{
    const SIZE: usize = size_of::<Self>();

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn empty() -> Self {
        Vault {
            bits: zero(),
            phantom: PhantomData,
        }
    }

    #[inline(always)]
    #[expect(clippy::needless_pass_by_value)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn close(mut key: Key, value: T) -> Self {
        const { assert!(U::BITS >= M) }
        key.push(value.encode());

        Vault {
            bits: key.cast(),
            phantom: PhantomData,
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn open(self, key: Key) -> Option<T> {
        const { assert!(U::BITS >= M) }

        if self.bits == zero() {
            return None;
        }

        let mut bits = self.bits.convert::<Key>().assume();
        let encoded: T::Bits = bits.pop();
        if bits == key.slice(..(U::BITS - M)) {
            Some(Binary::decode(encoded))
        } else {
            None
        }
    }
}

/// A generic memoization cache.
#[derive(Debug, Deref, DerefMut)]
pub struct Cache<T: Binary, U: Unsigned = <<T as Binary>::Bits as Int>::Repr> {
    #[deref(forward)]
    #[deref_mut(forward)]
    data: HugeSeq<Atomic<Vault<T, U>>>,
}

#[cfg(test)]
impl<T, U, R, const M: u32> Arbitrary for Cache<T, U>
where
    T: Arbitrary + Binary<Bits = Bits<R, M>>,
    U: Unsigned,
    R: Unsigned,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (hash_map(any::<Key>(), any::<T>(), ..32), ..128usize)
            .prop_map(|(map, size)| {
                let data: HugeSeq<Atomic<Vault<T, U>>> =
                    HugeSeq::zeroed(size * Vault::<T, U>::SIZE);

                if size > 0 {
                    for (k, v) in map {
                        data[k].store(Vault::close(k, v), Ordering::Relaxed);
                    }
                }

                Cache { data }
            })
            .boxed()
    }
}

impl<T, U, R, const M: u32> Cache<T, U>
where
    T: Binary<Bits = Bits<R, M>>,
    U: Unsigned,
    R: Unsigned,
{
    /// Allocates up to `size` bytes of cache space.
    #[inline(always)]
    pub fn new(size: usize) -> Self {
        Self {
            data: HugeSeq::zeroed(size / Vault::<T, U>::SIZE),
        }
    }

    /// Resizes the cache space to up to `size` bytes.
    #[inline(always)]
    pub fn resize(&mut self, size: usize) {
        self.data.zeroed_in_place(size / Vault::<T, U>::SIZE);
    }

    /// Instructs the CPU to load the slot associated with `key`.
    #[inline(always)]
    pub fn prefetch(&self, key: Key) {
        if !self.data.is_empty() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{_MM_HINT_ET0, _mm_prefetch};
                let ptr = &raw const self.data[key];
                _mm_prefetch(ptr.cast(), _MM_HINT_ET0);
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::{_PREFETCH_LOCALITY0, _PREFETCH_WRITE, _prefetch};
                let ptr = &raw const self.data[key];
                _prefetch(ptr.cast(), _PREFETCH_WRITE, _PREFETCH_LOCALITY0);
            }
        }
    }

    /// Stores a value in the slot associated with `key`.
    #[inline(always)]
    pub fn store(&self, key: Key, value: T) {
        if !self.data.is_empty() {
            self.data[key].store(Vault::close(key, value), Ordering::Relaxed);
        }
    }

    /// Loads the value from the slot associated with `key`.
    #[inline(always)]
    pub fn load(&self, key: Key) -> Option<T> {
        if !self.data.is_empty() {
            self.data[key].load(Ordering::Relaxed).open(key)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    type MockVault = Vault<u8, u64>;
    type MockCache = Cache<u8, u64>;

    #[proptest]
    fn opening_vault_with_correct_key_succeeds(k: Key, v: u8) {
        assert_eq!(MockVault::close(k, v).open(k), Some(v));
    }

    #[proptest]
    fn opening_vault_with_wrong_key_fails(k: Key, #[filter(#l != #k)] l: Key, v: u8) {
        assert_eq!(MockVault::close(k, v).open(l), None);
    }

    #[proptest]
    fn cache_allocates_up_to_size(#[strategy(..1024usize)] s: usize) {
        assert!(MockCache::new(s).len() * size_of::<MockVault>() <= s);
    }

    #[proptest]
    fn cache_resizes_up_to_size(mut c: MockCache, #[strategy(..1024usize)] s: usize) {
        c.resize(s);
        assert!(c.len() * size_of::<MockVault>() <= s);
    }

    #[proptest]
    fn load_does_nothing_if_capacity_is_zero(k: Key) {
        assert_eq!(MockCache::new(0).load(k), None);
    }

    #[proptest]
    fn load_returns_none_if_slot_is_empty(
        #[by_ref]
        #[filter(!#c.is_empty())]
        mut c: MockCache,
        k: Key,
    ) {
        c[k] = zero();
        assert_eq!(c.load(k), None);
    }

    #[proptest]
    fn load_returns_none_if_key_does_not_match(
        #[by_ref]
        #[filter(!#c.is_empty())]
        mut c: MockCache,
        k: Key,
        v: u8,
    ) {
        *c[k].get_mut() = Vault::close(!k, v);
        assert_eq!(c.load(k), None);
    }

    #[proptest]
    fn load_returns_some_if_key_matches(
        #[by_ref]
        #[filter(!#c.is_empty())]
        mut c: MockCache,
        k: Key,
        v: u8,
    ) {
        *c[k].get_mut() = Vault::close(k, v);
        assert_eq!(c.load(k), Some(v));
    }

    #[proptest]
    fn set_does_nothing_if_capacity_is_zero(k: Key, v: u8) {
        MockCache::new(0).store(k, v);
    }

    #[proptest]
    fn set_stores_value_if_slot_is_empty(
        #[by_ref]
        #[filter(!#c.is_empty())]
        mut c: MockCache,
        k: Key,
        v: u8,
    ) {
        c[k] = zero();
        c.store(k, v);
        assert_eq!(c.load(k), Some(v));
    }

    #[proptest]
    fn set_replaces_value_if_one_exists(
        #[by_ref]
        #[filter(!#c.is_empty())]
        mut c: MockCache,
        k: Key,
        u: u8,
        l: Key,
        v: u8,
    ) {
        *c[k].get_mut() = Vault::close(l, v);
        c.store(k, u);
        assert_eq!(c.load(k), Some(u));
    }
}

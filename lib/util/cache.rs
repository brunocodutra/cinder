use crate::util::*;
use bytemuck::{Pod, Zeroable};
use core::slice;
use derive_more::with_trait::{Debug, Deref, DerefMut};
use std::ops::{Div, Index, IndexMut};
use std::{cmp::Ordering, marker::PhantomData, sync::atomic::Ordering::Relaxed};

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

/// Trait for types that are worth storing in a [`Vault`].
pub const trait Valuable {
    type Worth: PartialOrd;

    fn worth(&self) -> Self::Worth;
}

/// A checksum-guarded container.
#[derive(Debug)]
#[debug("Vault({bits:?})")]
#[repr(transparent)]
pub struct Vault<T: Valuable + Binary, U: Unsigned> {
    bits: U,
    phantom: PhantomData<T>,
}

unsafe impl<T: Valuable + Binary, U: Unsigned> Zeroable for Vault<T, U> {}
unsafe impl<T: Valuable + Binary, U: Unsigned> Pod for Vault<T, U> {}

impl<T: Valuable + Binary, U: Unsigned> Copy for Vault<T, U> {}

impl<T: Valuable + Binary, U: Unsigned> Clone for Vault<T, U> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, U, R, const N: u32> PartialEq for Vault<T, U>
where
    T: Valuable + Binary<Bits = Bits<R, N>>,
    U: Unsigned,
    R: Unsigned,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return self.bits == other.bits;
        }

        let a = T::decode(self.bits.convert::<Key>().assume().pop());
        let b = T::decode(other.bits.convert::<Key>().assume().pop());
        a.worth().eq(&b.worth())
    }
}

impl<T, U, R, const N: u32> PartialOrd for Vault<T, U>
where
    T: Valuable + Binary<Bits = Bits<R, N>>,
    U: Unsigned,
    R: Unsigned,
{
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self.is_empty(), other.is_empty()) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (false, false) => {
                let a = T::decode(self.bits.convert::<Key>().assume().pop());
                let b = T::decode(other.bits.convert::<Key>().assume().pop());
                a.worth().partial_cmp(&b.worth())
            }
        }
    }
}

impl<T, U, R, const N: u32> Vault<T, U>
where
    T: Valuable + Binary<Bits = Bits<R, N>>,
    U: Unsigned,
    R: Unsigned,
{
    const SIZE: usize = size_of::<Self>();

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.bits == zero()
    }

    #[inline(always)]
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
        const { assert!(U::BITS >= N) }
        key.push(value.encode());

        Vault {
            bits: key.cast(),
            phantom: PhantomData,
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn open(self, key: Key) -> Option<T> {
        const { assert!(U::BITS >= N) }

        if self.is_empty() {
            return None;
        }

        let mut bits = self.bits.convert::<Key>().assume();
        let encoded: T::Bits = bits.pop();
        if bits == key.slice(..(U::BITS - N)) {
            Some(Binary::decode(encoded))
        } else {
            None
        }
    }

    /// Whether this key matches the lock.
    #[inline(always)]
    pub fn matches(&self, key: Key) -> bool {
        !self.is_empty() && (self.bits >> N.cast()) == key.slice(..(U::BITS - N)).cast()
    }
}

/// A [`Cache`] backed by [`HugePage`].
pub type HugeCache<T, U, const C: usize> = Cache<T, U, HugePages<Atomic<Vault<T, U>>>, C>;

/// A generic memoization cache.
#[derive(Debug, Deref, DerefMut)]
pub struct Cache<T: Valuable + Binary, U: Unsigned, M: Memory<Atomic<Vault<T, U>>>, const C: usize>
{
    #[deref(forward)]
    #[deref_mut(forward)]
    data: M,
    phantom: PhantomData<Vault<T, U>>,
}

#[cfg(test)]
impl<T, U, R, const N: u32, const C: usize> Arbitrary for HugeCache<T, U, C>
where
    T: Arbitrary + Valuable + Binary<Bits = Bits<R, N>>,
    U: Unsigned,
    R: Unsigned,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    #[expect(clippy::iter_over_hash_type)]
    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        (hash_map(any::<Key>(), any::<T>(), ..32), ..128usize)
            .prop_map(|(map, size)| {
                let cache = Self {
                    data: HugePages::zeroed(size * Vault::<T, U>::SIZE),
                    phantom: PhantomData,
                };

                if size > 0 {
                    for (k, v) in map {
                        cache.data[k].store(Vault::close(k, v), Relaxed);
                    }
                }

                cache
            })
            .boxed()
    }
}

impl<T, U, R, const N: u32, const C: usize> HugeCache<T, U, C>
where
    T: Valuable + Binary<Bits = Bits<R, N>>,
    U: Unsigned,
    R: Unsigned,
{
    /// Allocates up to `size` bytes of cache space.
    #[inline(always)]
    pub fn new(size: usize) -> Self {
        Self {
            data: HugePages::zeroed(size.div(Vault::<T, U>::SIZE).cast()),
            phantom: PhantomData,
        }
    }
}

impl<T, U, R, M, const N: u32, const C: usize> Cache<T, U, M, C>
where
    T: Valuable + Binary<Bits = Bits<R, N>>,
    U: Unsigned,
    M: Memory<Atomic<Vault<T, U>>> + Deref<Target = [Atomic<Vault<T, U>>]>,
    R: Unsigned,
{
    #[inline(always)]
    pub fn clusters(&self) -> &[[Atomic<Vault<T, U>>; C]] {
        const { assert!(N > 0) }
        unsafe { slice::from_raw_parts(self.data.as_ptr().cast(), self.len() / C) }
    }

    /// Resizes the cache space to up to `size` bytes.
    #[inline(always)]
    pub fn resize(&mut self, size: usize) {
        self.data
            .zeroed_in_place(size.div(Vault::<T, U>::SIZE).cast());
    }

    /// Instructs the CPU to load the slot associated with `key`.
    #[inline(always)]
    pub fn prefetch(&self, key: Key) {
        let clusters = self.clusters();

        if !clusters.is_empty() {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{_MM_HINT_ET0, _mm_prefetch};
                let ptr = &raw const clusters[key];
                _mm_prefetch(ptr.cast(), _MM_HINT_ET0);
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::{_PREFETCH_LOCALITY0, _PREFETCH_WRITE, _prefetch};
                let ptr = &raw const clusters[key];
                _prefetch(ptr.cast(), _PREFETCH_WRITE, _PREFETCH_LOCALITY0);
            }
        }
    }

    /// Stores a value in the slot associated with `key`.
    #[inline(always)]
    pub fn store(&self, key: Key, value: T) {
        const { assert!(size_of::<T::Worth>() > 0 || C == 1) }
        let clusters = self.clusters();
        if !clusters.is_empty() {
            if size_of::<T::Worth>() == 0 {
                clusters[key][0].store(Vault::close(key, value), Relaxed);
            } else {
                let mut cluster = clusters[key].iter().map(|s| s.load(Relaxed)).enumerate();
                let (mut idx, mut disposable) = cluster.next().assume();
                if !disposable.is_empty() && !disposable.matches(key) {
                    for (i, vault) in cluster {
                        if vault.matches(key) || vault.is_empty() {
                            (idx, disposable) = (i, vault);
                            break;
                        } else if vault < disposable {
                            (idx, disposable) = (i, vault);
                        }
                    }
                }

                let vault = Vault::close(key, value);
                if !disposable.matches(key) || vault >= disposable {
                    clusters[key][idx].store(vault, Relaxed);
                }
            }
        }
    }

    /// Loads the value from the slot associated with `key`.
    #[inline(always)]
    pub fn load(&self, key: Key) -> Option<T> {
        let clusters = self.clusters();
        if !clusters.is_empty() {
            for slot in &clusters[key] {
                if let Some(entry) = slot.load(Relaxed).open(key) {
                    return Some(entry);
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::{Arbitrary, proptest};

    #[derive(Debug, Copy, Hash, Arbitrary)]
    #[derive_const(Clone, Eq, PartialEq)]
    struct MockValuable(u8);

    impl Valuable for MockValuable {
        type Worth = u8;

        fn worth(&self) -> Self::Worth {
            self.0
        }
    }

    impl Binary for MockValuable {
        type Bits = <u8 as Binary>::Bits;

        fn encode(&self) -> Self::Bits {
            self.0.encode()
        }

        fn decode(bits: Self::Bits) -> Self {
            Self(u8::decode(bits))
        }
    }

    type MockVault = Vault<MockValuable, u64>;
    type MockCache = HugeCache<MockValuable, u64, 3>;

    #[proptest]
    fn opening_vault_with_correct_key_succeeds(k: Key, v: MockValuable) {
        assert_eq!(MockVault::close(k, v).open(k), Some(v));
    }

    #[proptest]
    fn opening_vault_with_wrong_key_fails(k: Key, #[filter(#l != #k)] l: Key, v: MockValuable) {
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
        v: MockValuable,
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
        v: MockValuable,
    ) {
        *c[k].get_mut() = Vault::close(k, v);
        assert_eq!(c.load(k), Some(v));
    }

    #[proptest]
    fn set_does_nothing_if_capacity_is_zero(k: Key, v: MockValuable) {
        MockCache::new(0).store(k, v);
    }

    #[proptest]
    fn set_stores_value_if_slot_is_empty(
        #[by_ref]
        #[filter(!#c.is_empty())]
        mut c: MockCache,
        k: Key,
        v: MockValuable,
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
        u: MockValuable,
        l: Key,
        v: MockValuable,
    ) {
        *c[k].get_mut() = Vault::close(l, v);
        c.store(k, u);
        assert_eq!(c.load(k), Some(u));
    }
}

use crate::util::{Assume, Binary, Bits, Int, Slice, Unsigned};
use atomic::Atomic;
use bytemuck::Zeroable;
use derive_more::with_trait::{Debug, Deref, DerefMut};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::{mem::size_of, sync::atomic::Ordering};

#[cfg(test)]
use proptest::{collection::*, prelude::*};

/// The key to a [`Padlock`].
pub type Key = Bits<u64, 64>;

impl<T> Index<Key> for [T] {
    type Output = T;

    #[inline(always)]
    fn index(&self, key: Key) -> &Self::Output {
        let idx = ((key.cast::<u128>() * self.len().cast::<u128>()) >> 64) as usize;
        self.get(idx).assume()
    }
}

impl<T> IndexMut<Key> for [T] {
    #[inline(always)]
    fn index_mut(&mut self, key: Key) -> &mut Self::Output {
        let idx = ((key.cast::<u128>() * self.len().cast::<u128>()) >> 64) as usize;
        self.get_mut(idx).assume()
    }
}

/// A checksum-guarded container.
#[derive(Debug)]
#[repr(transparent)]
pub struct Padlock<T: Binary, U: Binary> {
    bits: U::Bits,
    phantom: PhantomData<T>,
}

impl<T, U, R, const M: u32, const N: u32> Padlock<T, U>
where
    T: Binary<Bits = Bits<R, M>>,
    U: Unsigned + Binary<Bits = Bits<U, N>>,
    R: Unsigned + Binary,
{
    #[inline(always)]
    pub fn close(key: Key, value: T) -> Self {
        let mut bits = Bits::new(key.cast());
        bits.push(value.encode());

        Padlock {
            bits,
            phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn open(mut self, key: Key) -> Option<T> {
        const { assert!(N >= M) }

        let bits: T::Bits = self.bits.pop();
        if self.bits.get() == key.slice(..(N - M)).cast() {
            Some(Binary::decode(bits))
        } else {
            None
        }
    }
}

impl<T, U, R, const M: u32, const N: u32> Binary for Padlock<T, U>
where
    T: Binary<Bits = Bits<R, M>>,
    U: Unsigned + Binary<Bits = Bits<U, N>>,
    R: Unsigned + Binary,
{
    type Bits = Bits<U, N>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        self.bits
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        Padlock {
            bits,
            phantom: PhantomData,
        }
    }
}

#[derive(Debug)]
#[repr(transparent)]
pub struct Vault<T: Binary<Bits: Int<Repr: Binary>>, U: Binary = <<T as Binary>::Bits as Int>::Repr>
where
    Option<Padlock<T, U>>: Binary,
{
    #[allow(clippy::type_complexity)]
    slot: Atomic<<Option<Padlock<T, U>> as Binary>::Bits>,
}

unsafe impl<T: Binary<Bits: Int<Repr: Binary>>, U: Binary> Zeroable for Vault<T, U> where
    Option<Padlock<T, U>>: Binary
{
}

impl<T, U, R, const M: u32, const N: u32> Vault<T, U>
where
    T: Binary<Bits = Bits<R, M>>,
    U: Unsigned + Binary<Bits = Bits<U, N>>,
    R: Unsigned + Binary,
{
    #[inline(always)]
    pub fn clear(&self) {
        let bits = None::<Padlock<T, U>>.encode();
        self.slot.store(bits, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn store(&self, key: Key, value: T) {
        let bits = Some(Padlock::close(key, value)).encode();
        self.slot.store(bits, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn load(&self, key: Key) -> Option<T> {
        let bits = self.slot.load(Ordering::Relaxed);
        Option::<Padlock<T, U>>::decode(bits)?.open(key)
    }
}

/// A generic memoization data.
#[derive(Debug, Deref, DerefMut)]
#[debug("Memory")]
pub struct Memory<
    T: Binary<Bits: Int<Repr: Binary>>,
    U: Binary = <<T as Binary>::Bits as Int>::Repr,
> where
    Option<Padlock<T, U>>: Binary,
{
    #[deref(forward)]
    #[deref_mut(forward)]
    data: Slice<Vault<T, U>>,
}

#[cfg(test)]
impl<T, U, R, const M: u32, const N: u32> Arbitrary for Memory<T, U>
where
    T: Arbitrary + Binary<Bits = Bits<R, M>> + 'static,
    U: Unsigned + Binary<Bits = Bits<U, N>>,
    R: Unsigned + Binary,
{
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
        (hash_map(any::<Key>(), any::<T>(), ..32), ..128usize)
            .prop_map(|(map, size)| {
                let data: Slice<Vault<T, U>> = Slice::new(size * size_of::<T>()).unwrap();

                if size > 0 {
                    for (k, v) in map {
                        data[k].store(k, v);
                    }
                }

                Memory { data }
            })
            .boxed()
    }
}

impl<T, U, R, const M: u32, const N: u32> Memory<T, U>
where
    T: Binary<Bits = Bits<R, M>>,
    U: Unsigned + Binary<Bits = Bits<U, N>>,
    R: Unsigned + Binary,
{
    /// Constructs a memoization data of at most `size` many bytes.
    #[inline(always)]
    pub fn new(size: usize) -> Self {
        let cap = size / size_of::<Padlock<T, U>>();

        Memory {
            data: Slice::new(cap).unwrap(),
        }
    }

    /// The actual size of this memoization data in bytes.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.capacity() * size_of::<Padlock<T, U>>()
    }

    /// The actual size of this memoization data in number of entries.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Instructs the CPU to load the slot associated with `key`.
    #[inline(always)]
    pub fn prefetch(&self, key: Key) {
        if self.capacity() > 0 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::{_MM_HINT_ET0, _mm_prefetch};
                _mm_prefetch(&self.data[key] as *const _ as _, _MM_HINT_ET0);
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::{_PREFETCH_LOCALITY0, _PREFETCH_WRITE, _prefetch};
                let ptr = &self.data[key] as *const _ as _;
                _prefetch(ptr, _PREFETCH_WRITE, _PREFETCH_LOCALITY0);
            }
        }
    }

    /// Stores a value in the slot associated with `key`.
    #[inline(always)]
    pub fn set(&self, key: Key, value: T) {
        if self.capacity() > 0 {
            self.data[key].store(key, value);
        }
    }

    /// Loads the value from the slot associated with `key`.
    #[inline(always)]
    pub fn get(&self, key: Key) -> Option<T> {
        if self.capacity() > 0 {
            self.data[key].load(key)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Assume;
    use std::fmt::Debug;
    use test_strategy::{Arbitrary, proptest};

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
    #[repr(u8)]
    enum Order {
        Less = 1,
        Equal = 2,
        Greater = 3,
    }

    unsafe impl Int for Order {
        type Repr = u8;
        const MIN: Self::Repr = 1;
        const MAX: Self::Repr = 3;
    }

    impl Binary for Order {
        type Bits = Bits<u8, 2>;

        fn encode(&self) -> Self::Bits {
            self.convert().assume()
        }

        fn decode(bits: Self::Bits) -> Self {
            bits.convert().assume()
        }
    }

    type MockPadlock = Padlock<Order, u64>;
    type MockMemory = Memory<Order, u64>;

    #[proptest]
    fn opening_vault_with_correct_key_succeeds(k: Key, v: Order) {
        assert_eq!(MockPadlock::close(k, v).open(k), Some(v));
    }

    #[proptest]
    fn opening_vault_with_wrong_key_fails(k: Key, #[filter(#l != #k)] l: Key, v: Order) {
        assert_eq!(MockPadlock::close(k, v).open(l), None);
    }

    #[proptest]
    fn input_size_is_an_upper_limit(#[strategy(..1024usize)] s: usize) {
        assert!(MockMemory::new(s).size() <= s);
    }

    #[proptest]
    fn capacity_returns_maximum_number_of_elements(m: MockMemory) {
        assert_eq!(m.size() / size_of::<MockPadlock>(), m.capacity());
    }

    #[proptest]
    fn get_does_nothing_if_capacity_is_zero(k: Key) {
        assert_eq!(MockMemory::new(0).get(k), None);
    }

    #[proptest]
    fn get_returns_none_if_slot_is_empty(
        #[by_ref]
        #[filter(#m.capacity() > 0)]
        m: MockMemory,
        k: Key,
    ) {
        m.data[k].clear();
        assert_eq!(m.get(k), None);
    }

    #[proptest]
    fn get_returns_none_if_key_does_not_match(
        #[by_ref]
        #[filter(#m.capacity() > 0)]
        m: MockMemory,
        k: Key,
        v: Order,
    ) {
        m.data[k].store(!k, v);
        assert_eq!(m.get(k), None);
    }

    #[proptest]
    fn get_returns_some_if_key_matches(
        #[by_ref]
        #[filter(#m.capacity() > 0)]
        m: MockMemory,
        k: Key,
        v: Order,
    ) {
        m.data[k].store(k, v);
        assert_eq!(m.get(k), Some(v));
    }

    #[proptest]
    fn set_does_nothing_if_capacity_is_zero(k: Key, v: Order) {
        MockMemory::new(0).set(k, v);
    }

    #[proptest]
    fn set_stores_value_if_slot_is_empty(
        #[by_ref]
        #[filter(#m.capacity() > 0)]
        m: MockMemory,
        k: Key,
        v: Order,
    ) {
        m.data[k].clear();
        m.set(k, v);
        assert_eq!(m.get(k), Some(v));
    }

    #[proptest]
    fn set_replaces_value_if_one_exists(
        #[by_ref]
        #[filter(#m.capacity() > 0)]
        m: MockMemory,
        k: Key,
        u: Order,
        l: Key,
        v: Order,
    ) {
        m.data[k].store(l, v);
        m.set(k, u);
        assert_eq!(m.get(k), Some(u));
    }
}

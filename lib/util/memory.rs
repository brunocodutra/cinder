use crate::util::{Assume, Binary, Bits, Integer, Slice, Unsigned};
use atomic::Atomic;
use bytemuck::Zeroable;
use derive_more::with_trait::{Debug, Deref, DerefMut};
use std::ops::{Index, IndexMut};
use std::{marker::PhantomData, mem::size_of, sync::atomic::Ordering};

#[cfg(test)]
use proptest::{collection::*, prelude::*};

type Key = Bits<u64, 64>;

/// A [`Memory`] slot.
#[derive(Debug)]
#[repr(transparent)]
pub struct Latch<T: Binary, U: Binary> {
    bits: U::Bits,
    phantom: PhantomData<T>,
}

impl<T, U, R, const M: u32, const N: u32> Latch<T, U>
where
    T: Binary<Bits = Bits<R, M>>,
    U: Unsigned + Binary<Bits = Bits<U, N>>,
    R: Unsigned + Binary,
{
    #[inline(always)]
    fn close(key: Key, value: T) -> Self {
        let mut bits = Bits::new(key.cast());
        bits.push(value.encode());

        Latch {
            bits,
            phantom: PhantomData,
        }
    }

    #[inline(always)]
    fn open(mut self, key: Key) -> Option<T> {
        const { assert!(N >= M) }

        let bits: T::Bits = self.bits.pop();
        if self.bits.get() == key.slice(..(N - M)).cast() {
            Some(Binary::decode(bits))
        } else {
            None
        }
    }
}

impl<T, U, R, const M: u32, const N: u32> Binary for Latch<T, U>
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
        Latch {
            bits,
            phantom: PhantomData,
        }
    }
}

#[derive(Debug, Deref, DerefMut)]
#[repr(transparent)]
struct Slot<T>(Atomic<T>);

unsafe impl<T: Zeroable> Zeroable for Slot<T> {}

/// A generic memoization data.
#[derive(Debug)]
#[debug("Memory")]
pub struct Memory<
    T: Binary<Bits: Integer<Repr: Binary>>,
    U: Binary = <<T as Binary>::Bits as Integer>::Repr,
> where
    Option<Latch<T, U>>: Binary,
{
    #[allow(clippy::type_complexity)]
    data: Slice<Slot<<Option<Latch<T, U>> as Binary>::Bits>>,
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
        hash_map(any::<Key>(), any::<T>(), ..32)
            .prop_map(|map| {
                let cap = map.len().next_power_of_two();

                #[allow(clippy::type_complexity)]
                let mut data: Slice<Slot<<Option<Latch<T, U>> as Binary>::Bits>> =
                    Slice::new(cap).unwrap();

                if cap > 0 {
                    for (k, v) in map {
                        *data[k].get_mut() = Some(Latch::<T, U>::close(k, v)).encode();
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
        let cap = (1 + size / 2).next_power_of_two() / size_of::<Latch<T, U>>();

        Memory {
            data: Slice::new(cap).unwrap(),
        }
    }

    /// The actual size of this memoization data in bytes.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.capacity() * size_of::<Latch<T, U>>()
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
            let bits = Some(Latch::close(key, value)).encode();
            self.data[key].store(bits, Ordering::Relaxed);
        }
    }

    /// Loads the value from the slot associated with `key`.
    #[inline(always)]
    pub fn get(&self, key: Key) -> Option<T> {
        if self.capacity() == 0 {
            return None;
        }

        let bits = self.data[key].load(Ordering::Relaxed);
        Option::<Latch<T, U>>::decode(bits)?.open(key)
    }

    /// Clears all entries.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl<T> Index<Key> for [Slot<T>] {
    type Output = Slot<T>;

    #[inline(always)]
    fn index(&self, key: Key) -> &Self::Output {
        let idx = match Key::BITS - self.len().trailing_zeros() {
            Key::BITS => 0,
            w => key.slice(w..).cast(),
        };

        self.get(idx).assume()
    }
}

impl<T> IndexMut<Key> for [Slot<T>] {
    #[inline(always)]
    fn index_mut(&mut self, key: Key) -> &mut Self::Output {
        let idx = match Key::BITS - self.len().trailing_zeros() {
            Key::BITS => 0,
            w => key.slice(w..).cast(),
        };

        self.get_mut(idx).assume()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::{Arbitrary, proptest};

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
    #[repr(u8)]
    enum Alignment {
        Left = 1,
        Center = 2,
        Right = 3,
    }

    unsafe impl Integer for Alignment {
        type Repr = u8;
        const MIN: Self::Repr = 1;
        const MAX: Self::Repr = 3;
    }

    impl Binary for Alignment {
        type Bits = Bits<u8, 2>;

        fn encode(&self) -> Self::Bits {
            self.convert().assume()
        }

        fn decode(bits: Self::Bits) -> Self {
            bits.convert().assume()
        }
    }

    type MockLatch = Latch<Alignment, u16>;
    type MockMemory = Memory<Alignment, u16>;

    #[proptest]
    fn input_size_is_an_upper_limit(#[strategy(..1024usize)] s: usize) {
        assert!(MockMemory::new(s).size() <= s);
    }

    #[proptest]
    fn size_is_exact_if_input_is_power_of_two(
        #[strategy(size_of::<MockLatch>().trailing_zeros()..=10)] bits: u32,
    ) {
        let size: usize = 1usize << bits;
        assert_eq!(MockMemory::new(size).size(), size);
    }

    #[proptest]
    fn capacity_returns_maximum_number_of_elements(m: MockMemory) {
        assert_eq!(m.size() / size_of::<MockLatch>(), m.capacity());
    }

    #[proptest]
    fn get_does_nothing_if_capacity_is_zero(k: Key) {
        assert_eq!(MockMemory::new(0).get(k), None);
    }

    #[proptest]
    fn get_returns_none_if_slot_is_empty(m: MockMemory, k: Key) {
        m.data[k].store(None::<MockLatch>.encode(), Ordering::Relaxed);
        assert_eq!(m.get(k), None);
    }

    #[proptest]
    fn get_returns_none_if_key_does_not_match(m: MockMemory, k: Key, v: Alignment) {
        let latch = Some(Latch::close(!k, v));
        m.data[k].store(latch.encode(), Ordering::Relaxed);
        assert_eq!(m.get(k), None);
    }

    #[proptest]
    fn get_returns_some_if_key_matches(m: MockMemory, k: Key, v: Alignment) {
        let latch = Some(Latch::close(k, v));
        m.data[k].store(latch.encode(), Ordering::Relaxed);
        assert_eq!(m.get(k), Some(v));
    }

    #[proptest]
    fn set_does_nothing_if_capacity_is_zero(k: Key, v: Alignment) {
        MockMemory::new(0).set(k, v);
    }

    #[proptest]
    fn set_stores_value_if_slot_is_empty(m: MockMemory, k: Key, v: Alignment) {
        m.data[k].store(None::<MockLatch>.encode(), Ordering::Relaxed);
        m.set(k, v);
        assert_eq!(m.get(k), Some(v));
    }

    #[proptest]
    fn set_replaces_value_if_one_exists(m: MockMemory, k: Key, u: Alignment, l: Key, v: Alignment) {
        let latch = Some(Latch::close(l, v));
        m.data[k].store(latch.encode(), Ordering::Relaxed);
        m.set(k, u);
        assert_eq!(m.get(k), Some(u));
    }
}

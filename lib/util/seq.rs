use crate::util::*;
use bytemuck::{NoUninit, Zeroable};
use derive_more::with_trait::Debug;
use std::hash::{Hash, Hasher};
use std::marker::{ConstParamTy, Destruct, PhantomData};
use std::mem::{ManuallyDrop, needs_drop};
use std::ops::{Deref, DerefMut};
use std::{ptr, slice};

#[cfg(test)]
use proptest::{collection::*, prelude::*};

/// A const-allocated collection of raw bytes.
pub type ConstBytes<const S: usize> = Bytes<ConstMemory<S>>;

/// A collection of raw bytes.
#[derive(Debug, Copy, Hash, ConstParamTy)]
#[derive_const(Default, Clone, Eq, PartialEq)]
pub struct Bytes<M> {
    len: <ConstCapacity as Capacity>::Usize,
    mem: M,
}

impl<T: NoUninit, const N: usize, const S: usize> const From<[T; N]> for Bytes<ConstMemory<S>> {
    #[inline(always)]
    fn from(data: [T; N]) -> Self {
        const { assert!(!needs_drop::<T>()) }

        Bytes {
            len: N.convert().assume(),
            mem: data.into(),
        }
    }
}

/// A [`Seq`] backed by [`ConstMemory`].
pub type ConstSeq<T, const S: usize> = Seq<T, ConstMemory<S>>;

/// A [`Seq`] backed by [`StaticMemory`].
pub type StaticSeq<T, const N: usize> = Seq<T, StaticMemory<T, N>>;

/// A const-allocated sequence of objects of type `T`.
#[derive(Debug)]
#[debug("Seq({:?})", self.deref())]
#[debug(bounds(T: Debug))]
pub struct Seq<T, M: Memory<T, Capacity = ConstCapacity>> {
    bytes: Bytes<M>,
    phantom: PhantomData<T>,
}

#[cfg(test)]
impl<T: Arbitrary + 'static, const N: usize> Arbitrary for StaticSeq<T, N> {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        vec(any::<T>(), 0..=N)
            .prop_map(move |items| {
                let mut seq = Self::new();

                for i in items {
                    seq.push(i);
                }

                seq
            })
            .boxed()
    }
}

const impl<T: Zeroable, M: [const] Memory<T, Capacity = ConstCapacity>> Seq<T, M> {
    /// Allocates a new [`Seq`] with `len` zero-initialized objects of type `T`.
    #[inline(always)]
    pub fn zeroed(len: usize) -> Self {
        let mem = M::zeroed(ConstCapacity);
        (len <= mem.as_ref().len()).assume();

        Seq {
            phantom: PhantomData,
            bytes: Bytes {
                len: len.cast(),
                mem,
            },
        }
    }
}

const impl<T, M: [const] Memory<T, Capacity = ConstCapacity>> Seq<T, M> {
    /// Allocates a new [`Seq`] for up to `capacity` objects of type `T`.
    #[inline(always)]
    pub fn new() -> Self {
        Seq {
            phantom: PhantomData,
            bytes: Bytes {
                len: zero(),
                mem: M::uninit(ConstCapacity),
            },
        }
    }

    /// Reifies raw [`Bytes`] into a sequence of `T`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `bytes` contains objects of type `T`.
    #[inline(always)]
    pub unsafe fn reify(bytes: Bytes<M>) -> Self {
        Seq {
            phantom: PhantomData,
            bytes,
        }
    }

    /// The total capacity of the underlying [`Memory`].
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.bytes.mem.as_ref().len()
    }

    /// This sequence's current length.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.bytes.len.cast()
    }

    /// Whether this sequence's current length is zero.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Pushes a new item into this sequence.
    #[inline(always)]
    pub fn push(&mut self, item: T) {
        let idx = self.len();
        let items = self.bytes.mem.as_mut();
        items.get_mut(idx).assume().write(item);
        self.bytes.len += ones::<<ConstCapacity as Capacity>::Usize>(1);
    }

    /// Pops an item from the back of this sequence.
    #[inline(always)]
    pub fn pop(&mut self) -> Option<T> {
        if self.bytes.len == zero::<<ConstCapacity as Capacity>::Usize>() {
            return None;
        }

        self.bytes.len -= ones::<<ConstCapacity as Capacity>::Usize>(1);
        let idx = self.len();
        let items = self.bytes.mem.as_mut();
        unsafe { Some(items.get(idx).assume().assume_init_read()) }
    }

    /// Truncates this sequence to the prefix of length `len`.
    #[inline(always)]
    pub fn truncate(&mut self, len: usize)
    where
        T: [const] Destruct,
    {
        let tail = self.len().min(len);
        unsafe { ptr::drop_in_place(self.get_mut(tail..).assume()) };
        self.bytes.len = tail.cast();
    }
}

impl<T, M> const Drop for Seq<T, M>
where
    T: [const] Destruct,
    M: [const] Memory<T, Capacity = ConstCapacity>,
{
    #[inline(always)]
    fn drop(&mut self) {
        self.truncate(zero());
    }
}

impl<T, M> const Default for Seq<T, M>
where
    M: [const] Memory<T, Capacity = ConstCapacity>,
{
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, M> const Clone for Seq<T, M>
where
    T: Copy,
    M: [const] Memory<T, Capacity = ConstCapacity>,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        let mut seq = Self::new();
        seq.bytes.len = self.bytes.len;
        let prefix = seq.len();
        let src = self.bytes.mem.as_ref().get(..prefix).assume();
        let dst = seq.bytes.mem.as_mut().get_mut(..prefix).assume();
        dst.copy_from_slice(src);
        seq
    }
}

impl<T, M> const Eq for Seq<T, M>
where
    T: [const] Eq,
    M: [const] Memory<T, Capacity = ConstCapacity>,
{
}

impl<T, M> const PartialEq for Seq<T, M>
where
    T: [const] PartialEq,
    M: [const] Memory<T, Capacity = ConstCapacity>,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<T, M> Hash for Seq<T, M>
where
    T: Hash,
    M: Memory<T, Capacity = ConstCapacity>,
{
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}

impl<T, M> const Deref for Seq<T, M>
where
    M: [const] Memory<T, Capacity = ConstCapacity>,
{
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        let len = self.bytes.len.cast();
        let uninit = self.bytes.mem.as_ref();
        unsafe { uninit.get_unchecked(..len).assume_init_ref() }
    }
}

impl<T, M> const DerefMut for Seq<T, M>
where
    M: [const] Memory<T, Capacity = ConstCapacity>,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let len = self.bytes.len.cast();
        let uninit = self.bytes.mem.as_mut();
        unsafe { uninit.get_unchecked_mut(..len).assume_init_mut() }
    }
}

impl<T, M: Memory<T, Capacity = ConstCapacity>> Extend<T> for Seq<T, M> {
    #[inline(always)]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for i in iter {
            self.push(i);
        }
    }
}

impl<T, M: Memory<T, Capacity = ConstCapacity>> FromIterator<T> for Seq<T, M> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut seq = Self::new();
        seq.extend(iter);
        seq
    }
}

impl<'a, T, M: Memory<T, Capacity = ConstCapacity>> IntoIterator for &'a Seq<T, M> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, M: Memory<T, Capacity = ConstCapacity>> IntoIterator for &'a mut Seq<T, M> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, M: Memory<T, Capacity = ConstCapacity>> IntoIterator for Seq<T, M> {
    type Item = T;
    type IntoIter = Iter<T, M>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

#[derive(Debug)]
pub struct Iter<T, M: Memory<T, Capacity = ConstCapacity>> {
    cursor: <ConstCapacity as Capacity>::Usize,
    seq: ManuallyDrop<Seq<T, M>>,
}

const impl<T, M: [const] Memory<T, Capacity = ConstCapacity>> Iter<T, M> {
    #[inline(always)]
    fn new(seq: Seq<T, M>) -> Self {
        Iter {
            seq: ManuallyDrop::new(seq),
            cursor: zero(),
        }
    }
}

impl<T, M> const Drop for Iter<T, M>
where
    T: [const] Destruct,
    M: [const] Memory<T, Capacity = ConstCapacity>,
{
    #[inline(always)]
    fn drop(&mut self) {
        const { assert!(!needs_drop::<M>()) }
        self.seq.truncate(self.cursor.cast());
    }
}

impl<T, M: Memory<T, Capacity = ConstCapacity>> ExactSizeIterator for Iter<T, M> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.seq.len().cast::<usize>() - self.cursor.cast::<usize>()
    }
}

impl<T, M: Memory<T, Capacity = ConstCapacity>> Iterator for Iter<T, M> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            None
        } else {
            let items = self.seq.bytes.mem.as_mut();
            let idx = self.cursor.cast::<usize>();
            let item = unsafe { items.get(idx).assume().assume_init_read() };
            self.cursor += ones::<<ConstCapacity as Capacity>::Usize>(1);
            Some(item)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use test_strategy::proptest;

    #[derive(Debug)]
    struct IncOnDrop<'a>(&'a AtomicUsize);

    impl Drop for IncOnDrop<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn capacity_returns_maximum_number_of_elements() {
        assert_eq!(StaticSeq::<u32, 16>::new().capacity(), 16);
        assert_eq!(ConstSeq::<u32, 64>::new().capacity(), 16);
    }

    #[test]
    fn empty_by_default() {
        assert!(StaticSeq::<u32, 16>::default().is_empty());
        assert!(ConstSeq::<u32, 64>::default().is_empty());
    }

    #[proptest]
    fn can_be_zero_initialized(#[strategy(..10usize)] n: usize) {
        assert!(StaticSeq::<u32, 16>::zeroed(n).iter().all(|x| *x == 0));
        assert!(ConstSeq::<u32, 64>::zeroed(n).iter().all(|x| *x == 0));
    }

    #[proptest]
    fn len_is_capacity_when_zero_initialized(#[strategy(..10usize)] n: usize) {
        assert_eq!(StaticSeq::<u32, 16>::zeroed(n).len(), n);
        assert_eq!(ConstSeq::<u32, 64>::zeroed(n).len(), n);
    }

    #[proptest]
    fn can_be_cloned(seq: StaticSeq<u32, 16>) {
        assert_eq!(seq.clone(), seq);
    }

    #[proptest]
    fn push_appends_element(
        #[filter(#seq.capacity() > #seq.len())] mut seq: StaticSeq<u32, 16>,
        e: u32,
    ) {
        let len = seq.len();
        seq.push(e);

        assert_eq!(seq.len(), len + 1);
        assert_eq!(seq.last().copied(), Some(e));
    }

    #[proptest]
    fn pop_removes_last_element(mut seq: StaticSeq<u32, 16>) {
        let len = seq.len();
        assert_eq!(seq.last().copied(), seq.pop());
        assert_eq!(seq.len(), len.saturating_sub(1));
    }

    #[proptest]
    fn truncate_resets_length(mut seq: StaticSeq<u32, 16>, n: usize) {
        let len = seq.len();
        seq.truncate(n);
        assert_eq!(seq.len(), len.min(n));
    }

    #[proptest]
    fn truncate_drops_excess_elements(#[strategy(..10usize)] m: usize, n: usize) {
        let counter = AtomicUsize::new(0);
        let mut seq = StaticSeq::<IncOnDrop<'_>, 16>::default();

        for _ in 0..m {
            seq.push(IncOnDrop(&counter));
        }

        seq.truncate(n);
        assert_eq!(counter.load(Ordering::SeqCst), m.max(n) - n);
    }

    #[proptest]
    fn drops_elements_when_dropped(#[strategy(..10usize)] n: usize) {
        let counter = AtomicUsize::new(0);
        let mut seq = StaticSeq::<IncOnDrop<'_>, 16>::default();

        for _ in 0..n {
            seq.push(IncOnDrop(&counter));
        }

        drop(seq);
        assert_eq!(counter.load(Ordering::SeqCst), n);
    }

    #[proptest]
    fn can_be_consumed_as_iterator(seq: StaticSeq<u32, 16>) {
        assert_eq!(&*Vec::from_iter(seq.clone().into_iter()), &*seq);
    }

    #[proptest]
    fn drops_elements_when_consumed_as_iterator(
        #[strategy(..10usize)] m: usize,
        #[strategy(..10usize)] n: usize,
    ) {
        let counter = AtomicUsize::new(0);
        let mut seq = StaticSeq::<IncOnDrop<'_>, 16>::default();

        for _ in 0..m {
            seq.push(IncOnDrop(&counter));
        }

        for _ in seq.into_iter().take(n.cast()) {}
        assert_eq!(counter.load(Ordering::SeqCst), m);
    }

    #[proptest]
    fn can_be_extended(
        #[filter(#seq.capacity() > #seq.len())] mut seq: StaticSeq<u32, 16>,
        #[strategy(vec(any::<u32>(), ..#seq.capacity() - #seq.len()))] v: Vec<u32>,
    ) {
        let len = seq.len().cast::<usize>();
        seq.extend(v.iter().copied());
        assert_eq!(&seq[len..], &*v);
    }
}

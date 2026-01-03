use crate::util::*;
use bytemuck::{NoUninit, Zeroable};
use derive_more::with_trait::Debug;
use std::hash::{Hash, Hasher};
use std::marker::{ConstParamTy, Destruct, PhantomData};
use std::mem::{ManuallyDrop, MaybeUninit, needs_drop};
use std::ops::{Deref, DerefMut};
use std::{ptr, slice};

#[cfg(test)]
use proptest::{collection::*, prelude::*};

/// A const sequence of raw bytes.
pub type ConstBytes<const S: usize> = Bytes<ConstMemory<S>, <ConstMemory<S> as Memory<u8>>::Usize>;

/// A sequence of raw bytes.
#[derive(Debug, Copy, Hash, ConstParamTy)]
#[derive_const(Default, Clone, Eq, PartialEq)]
pub struct Bytes<M, U: Unsigned> {
    len: U,
    mem: M,
}

impl<T: NoUninit, const N: usize, const S: usize> const From<[T; N]>
    for Bytes<ConstMemory<S>, <ConstMemory<S> as Memory<T>>::Usize>
{
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

/// A [`Seq`] backed by [`DynamicMemory`].
pub type DynamicSeq<T> = Seq<T, DynamicMemory<T>>;

/// A [`Seq`] backed by [`HugePage`].
pub type HugeSeq<T> = Seq<T, HugePage<T>>;

/// A sequence of objects of type `T`.
#[derive(Debug)]
#[debug("Seq({:?})", self.deref())]
#[debug(bounds(T: Debug))]
pub struct Seq<T, M: Memory<T>> {
    bytes: Bytes<M, M::Usize>,
    phantom: PhantomData<T>,
}

#[cfg(test)]
impl<T: Arbitrary, M: Memory<T, Usize: From<u8>>> Arbitrary for Seq<T, M> {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with((): Self::Parameters) -> Self::Strategy {
        any::<u8>()
            .prop_flat_map(|cap| {
                vec(any::<T>(), 0..=cap.cast()).prop_map(move |items| {
                    let mut seq = Self::new(cap.into());

                    for i in items {
                        seq.push(i);
                    }

                    seq
                })
            })
            .boxed()
    }
}

impl<T: Zeroable, M: Memory<T>> Seq<T, M> {
    /// Allocates a new [`Seq`] with `len` zero-initialized objects of type `T`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn zeroed(len: M::Usize) -> Self {
        Seq {
            phantom: PhantomData,
            bytes: Bytes {
                len,
                mem: M::zeroed(len),
            },
        }
    }

    /// Reallocates a new [`Seq`] in-place with `len` zero-initialized objects of type `T`.
    ///
    /// This method guarantees existing memory is deallocated before reallocating.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn zeroed_in_place(&mut self, len: M::Usize) {
        unsafe { ptr::drop_in_place(self) }; // IMPORTANT: deallocate first
        unsafe { ptr::write(self, Self::zeroed(len)) };
    }
}

impl<T, M: Memory<T>> Seq<T, M> {
    /// Allocates a new [`Seq`] for up to `capacity` objects of type `T`.
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(capacity: M::Usize) -> Self {
        Seq {
            phantom: PhantomData,
            bytes: Bytes {
                len: zero(),
                mem: M::uninit(capacity),
            },
        }
    }

    /// Reifies raw [`Bytes`] into a sequence of `T`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee `bytes` contains objects of type `T`.
    #[inline(always)]
    pub const unsafe fn reify(bytes: Bytes<M, M::Usize>) -> Self {
        Seq {
            phantom: PhantomData,
            bytes,
        }
    }

    /// The total capacity of the underlying [`Memory`].
    #[inline(always)]
    pub const fn capacity(&mut self) -> M::Usize
    where
        M: [const] AsRef<[MaybeUninit<T>]>,
    {
        self.bytes.mem.as_ref().len().cast()
    }

    /// This sequence's current length.
    #[inline(always)]
    pub const fn len(&self) -> M::Usize {
        self.bytes.len
    }

    /// Pushes a new item into this sequence.
    #[inline(always)]
    pub const fn push(&mut self, item: T)
    where
        M: [const] AsRef<[MaybeUninit<T>]> + [const] AsMut<[MaybeUninit<T>]>,
        M::Usize: [const] Unsigned,
    {
        let idx = self.len().cast::<usize>();
        let items = self.bytes.mem.as_mut();
        items.get_mut(idx).assume().write(item);
        self.bytes.len += ones(1);
    }

    /// Pops an item from the back of this sequence.
    #[inline(always)]
    pub const fn pop(&mut self) -> Option<T>
    where
        M: [const] AsRef<[MaybeUninit<T>]> + [const] AsMut<[MaybeUninit<T>]>,
        M::Usize: [const] Unsigned,
    {
        if self.bytes.len == zero() {
            return None;
        }

        self.bytes.len -= ones(1);
        let idx = self.len().cast::<usize>();
        let items = self.bytes.mem.as_mut();
        unsafe { Some(items.get(idx).assume().assume_init_read()) }
    }

    /// Truncates this sequence to the prefix of length `len`.
    #[inline(always)]
    pub const fn truncate(&mut self, len: M::Usize)
    where
        T: [const] Destruct,
        M: [const] AsRef<[MaybeUninit<T>]> + [const] AsMut<[MaybeUninit<T>]>,
        M::Usize: [const] Unsigned,
    {
        let tail = self.bytes.len.min(len);
        unsafe { ptr::drop_in_place(self.get_mut(tail.cast::<usize>()..).assume()) };
        self.bytes.len = tail;
    }
}

impl<T, M: Memory<T>> Drop for Seq<T, M> {
    #[inline(always)]
    fn drop(&mut self) {
        self.truncate(zero());
    }
}

impl<T, M: Memory<T, Usize: [const] Unsigned> + [const] Default> const Default for Seq<T, M> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            bytes: Bytes::default(),
            phantom: PhantomData,
        }
    }
}

impl<T, M> const Clone for Seq<T, M>
where
    T: Copy,
    M: Memory<T, Usize: [const] Unsigned>
        + [const] Default
        + [const] AsRef<[MaybeUninit<T>]>
        + [const] AsMut<[MaybeUninit<T>]>,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        let mut seq = Self::default();
        seq.bytes.len = self.bytes.len;
        let prefix = seq.bytes.len.cast::<usize>();
        let src = self.bytes.mem.as_ref().get(..prefix).assume();
        let dst = seq.bytes.mem.as_mut().get_mut(..prefix).assume();
        dst.copy_from_slice(src);
        seq
    }
}

impl<T, M> const Eq for Seq<T, M>
where
    T: [const] Eq,
    M: Memory<T, Usize: [const] Unsigned> + [const] AsRef<[MaybeUninit<T>]>,
{
}

impl<T, M> const PartialEq for Seq<T, M>
where
    T: [const] PartialEq,
    M: Memory<T, Usize: [const] Unsigned> + [const] AsRef<[MaybeUninit<T>]>,
{
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<T: Hash, M: Memory<T>> Hash for Seq<T, M> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}

impl<T, M> const Deref for Seq<T, M>
where
    M: Memory<T, Usize: [const] Unsigned> + [const] AsRef<[MaybeUninit<T>]>,
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
    M: Memory<T, Usize: [const] Unsigned>
        + [const] AsRef<[MaybeUninit<T>]>
        + [const] AsMut<[MaybeUninit<T>]>,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        let len = self.bytes.len.cast();
        let uninit = self.bytes.mem.as_mut();
        unsafe { uninit.get_unchecked_mut(..len).assume_init_mut() }
    }
}

impl<T, M: Memory<T>> Extend<T> for Seq<T, M> {
    #[inline(always)]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for i in iter {
            self.push(i);
        }
    }
}

impl<T, M: Memory<T> + Default> FromIterator<T> for Seq<T, M> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut seq = Self::default();
        seq.extend(iter);
        seq
    }
}

impl<'a, T, M: Memory<T>> IntoIterator for &'a Seq<T, M> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, M: Memory<T>> IntoIterator for &'a mut Seq<T, M> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, M: Memory<T>> IntoIterator for Seq<T, M> {
    type Item = T;
    type IntoIter = Iter<T, M>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self)
    }
}

#[derive(Debug)]
pub struct Iter<T, M: Memory<T>> {
    cursor: M::Usize,
    seq: ManuallyDrop<Seq<T, M>>,
}

impl<T, M: Memory<T>> Iter<T, M> {
    #[inline(always)]
    const fn new(seq: Seq<T, M>) -> Self {
        Iter {
            seq: ManuallyDrop::new(seq),
            cursor: zero(),
        }
    }
}

impl<T, M: Memory<T>> Drop for Iter<T, M> {
    #[inline(always)]
    fn drop(&mut self) {
        const { assert!(!needs_drop::<M>()) }
        self.seq.truncate(self.cursor);
    }
}

impl<T, M: Memory<T>> ExactSizeIterator for Iter<T, M> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.seq.len().cast::<usize>() - self.cursor.cast::<usize>()
    }
}

impl<T, M: Memory<T>> Iterator for Iter<T, M> {
    type Item = T;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len() == 0 {
            None
        } else {
            let items = self.seq.bytes.mem.as_mut();
            let idx = self.cursor.cast::<usize>();
            let item = unsafe { items.get(idx).assume().assume_init_read() };
            self.cursor += ones(1);
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
    use std::sync::atomic::{AtomicU32, Ordering};
    use test_strategy::proptest;

    #[derive(Debug)]
    struct IncOnDrop<'a>(&'a AtomicU32);

    impl Drop for IncOnDrop<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[proptest]
    fn can_be_zero_initialized(#[strategy(..10usize)] n: usize) {
        let seq = DynamicSeq::<u32>::zeroed(n);
        assert!(seq.iter().all(|x| *x == 0));
    }

    #[proptest]
    fn can_be_reinitialized_in_place(
        #[strategy(..10usize)] m: usize,
        #[strategy(..10usize)] n: usize,
    ) {
        let mut seq = DynamicSeq::<u32>::zeroed(m);

        assert_eq!(seq.len(), m);
        seq.zeroed_in_place(n);
        assert_eq!(seq.len(), n);

        assert!(seq.iter().all(|x| *x == 0));
    }

    #[proptest]
    fn can_be_cloned(seq: StaticSeq<u32, 256>) {
        assert_eq!(seq.clone(), seq);
    }

    #[proptest]
    fn capacity_returns_maximum_number_of_elements(#[strategy(..10usize)] n: usize) {
        assert_eq!(ConstSeq::<u32, 64>::new(n.cast()).capacity(), 16);
        assert_eq!(StaticSeq::<u32, 16>::new(n.cast()).capacity(), 16);
        assert_eq!(DynamicSeq::<u32>::new(n).capacity(), n);
        assert_eq!(HugeSeq::<u32>::new(n).capacity(), n);
    }

    #[proptest]
    fn len_returns_current_number_of_elements(#[strategy(..10usize)] n: usize) {
        assert_eq!(ConstSeq::<u32, 64>::zeroed(n.cast()).len(), n.cast());
        assert_eq!(StaticSeq::<u32, 16>::zeroed(n.cast()).len(), n.cast());
        assert_eq!(DynamicSeq::<u32>::zeroed(n).len(), n);
        assert_eq!(HugeSeq::<u32>::zeroed(n).len(), n);
    }

    #[test]
    fn empty_by_default() {
        assert_eq!(ConstSeq::<u32, 64>::default().len(), 0);
        assert_eq!(StaticSeq::<u32, 16>::default().len(), 0);
    }

    #[proptest]
    fn push_appends_element(mut seq: StaticSeq<u32, 256>, e: u32) {
        let len = seq.len();
        seq.push(e);

        assert_eq!(seq.len(), len + 1);
        assert_eq!(seq.last().copied(), Some(e));
    }

    #[proptest]
    fn pop_removes_last_element(mut seq: StaticSeq<u32, 256>) {
        let len = seq.len();
        assert_eq!(seq.last().copied(), seq.pop());
        assert_eq!(seq.len(), len.saturating_sub(1));
    }

    #[proptest]
    fn truncate_resets_length(mut seq: StaticSeq<u32, 256>, n: u32) {
        let len = seq.len();
        seq.truncate(n);
        assert_eq!(seq.len(), len.min(n));
    }

    #[proptest]
    fn truncate_drops_excess_elements(#[strategy(..10u32)] m: u32, n: u32) {
        let counter = AtomicU32::new(0);
        let mut seq = StaticSeq::<IncOnDrop<'_>, 16>::default();

        for _ in 0..m {
            seq.push(IncOnDrop(&counter));
        }

        seq.truncate(n);
        assert_eq!(counter.load(Ordering::SeqCst), m.max(n) - n);
    }

    #[proptest]
    fn drops_elements_when_dropped(#[strategy(..10u32)] n: u32) {
        let counter = AtomicU32::new(0);
        let mut seq = StaticSeq::<IncOnDrop<'_>, 16>::default();

        for _ in 0..n {
            seq.push(IncOnDrop(&counter));
        }

        drop(seq);
        assert_eq!(counter.load(Ordering::SeqCst), n);
    }

    #[proptest]
    fn can_be_consumed_as_iterator(seq: StaticSeq<u32, 256>) {
        assert_eq!(&*Vec::from_iter(seq.clone().into_iter()), &*seq);
    }

    #[proptest]
    fn drops_elements_when_consumed_as_iterator(
        #[strategy(..10u32)] m: u32,
        #[strategy(..10u32)] n: u32,
    ) {
        let counter = AtomicU32::new(0);
        let mut seq = StaticSeq::<IncOnDrop<'_>, 16>::default();

        for _ in 0..m {
            seq.push(IncOnDrop(&counter));
        }

        for _ in seq.into_iter().take(n.cast()) {}
        assert_eq!(counter.load(Ordering::SeqCst), m);
    }

    #[proptest]
    fn can_be_extended(
        mut seq: StaticSeq<u32, 512>,
        #[strategy(vec(any::<u32>(), ..256))] v: Vec<u32>,
    ) {
        let len = seq.len().cast::<usize>();
        seq.extend(v.iter().copied());
        assert_eq!(&seq[len..], &*v);
    }
}

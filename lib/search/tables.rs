use crate::chess::Zobrist;
use crate::search::{Age, HashSize, Transposition, Value};
use crate::util::{Atomic, HugePages, Memory, Num, Prefetch, Vault};
use bytemuck::zeroed;
use derive_more::with_trait::{Debug, Deref, DerefMut};
use std::{cell::UnsafeCell, mem::MaybeUninit, ops::Shr, ptr, slice, sync::atomic::Ordering};

#[inline(always)]
const fn tt_size(size: HashSize) -> usize {
    size.get().max(1 << 18) - vt_size(size)
}

#[inline(always)]
const fn vt_size(size: HashSize) -> usize {
    size.get().max(1 << 18).shr(4) // ~6.25%
}

#[derive(Debug)]
#[debug("TranspositionTable({})", entries.len())]
pub struct TranspositionTable {
    entries: HugePages<Atomic<Vault<Transposition, u64>>>,
    age: Atomic<Age>,
}

impl TranspositionTable {
    #[inline(always)]
    pub fn new(size: HashSize) -> Self {
        Self {
            entries: HugePages::zeroed(tt_size(size).shr(3u32).cast()),
            age: zeroed(),
        }
    }

    #[inline(always)]
    pub fn resize(&mut self, size: HashSize) {
        self.entries.zeroed_in_place(tt_size(size).shr(3u32).cast());
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn store(&self, zobrist: Zobrist, mut new: Transposition) {
        new.age = self.age.load(Ordering::Relaxed);

        let slot = &self.entries[zobrist];
        let Some(old) = slot.load(Ordering::Relaxed).open(zobrist) else {
            return slot.store(Vault::close(zobrist, new), Ordering::Relaxed);
        };

        if new.age != old.age || new.depth >= old.depth - 4 {
            slot.store(Vault::close(zobrist, new), Ordering::Relaxed);
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn load(&self, zobrist: Zobrist) -> Option<Transposition> {
        self.entries[zobrist].load(Ordering::Relaxed).open(zobrist)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn prefetch(&self, zobrist: Zobrist) {
        ptr::from_ref(&self.entries[zobrist]).prefetch();
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn age(&mut self) {
        let age = self.age.get_mut();
        *age = Age::new(age.get().wrapping_add(1) & Age::MAX);
    }
}

impl const Deref for TranspositionTable {
    type Target = [UnsafeCell<MaybeUninit<u64>>];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.entries.as_ptr().cast(), self.entries.len()) }
    }
}

impl const DerefMut for TranspositionTable {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.entries.as_mut_ptr().cast(), self.entries.len()) }
    }
}

#[derive(Debug)]
#[debug("ValueTable({})", entries.len())]
pub struct ValueTable {
    entries: HugePages<Atomic<Vault<Value, u32>>>,
}

impl ValueTable {
    #[inline(always)]
    pub fn new(size: HashSize) -> Self {
        Self {
            entries: HugePages::zeroed(vt_size(size).shr(3u32).cast()),
        }
    }

    #[inline(always)]
    pub fn resize(&mut self, size: HashSize) {
        self.entries.zeroed_in_place(vt_size(size).shr(3u32).cast());
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn store(&self, key: Zobrist, value: Value) {
        self.entries[key].store(Vault::close(key, value), Ordering::Relaxed);
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn load(&self, key: Zobrist) -> Option<Value> {
        self.entries[key].load(Ordering::Relaxed).open(key)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn prefetch(&self, key: Zobrist) {
        ptr::from_ref(&self.entries[key]).prefetch();
    }
}

impl const Deref for ValueTable {
    type Target = [UnsafeCell<MaybeUninit<u32>>];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.entries.as_ptr().cast(), self.entries.len()) }
    }
}

impl const DerefMut for ValueTable {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.entries.as_mut_ptr().cast(), self.entries.len()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::zero;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn hash_size_is_split_between_tables(s: HashSize) {
        assert!(s.get().max(1 << 18) >= tt_size(s) + vt_size(s));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_allocates_up_to_tt_size(s: HashSize) {
        assert!(tt_size(s) >= TranspositionTable::new(s).len() * size_of::<u64>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_resizes_up_to_tt_size(s: HashSize, t: HashSize) {
        let mut tt = TranspositionTable::new(s);
        tt.resize(t);
        assert!(tt_size(t) >= tt.len() * size_of::<u64>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_load_returns_none_if_slot_is_empty(s: HashSize, k: Zobrist) {
        let mut tt = TranspositionTable::new(s);
        tt.entries[k] = zero();
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_load_returns_none_if_key_does_not_match(s: HashSize, k: Zobrist, v: Transposition) {
        let mut tt = TranspositionTable::new(s);
        *tt.entries[k].get_mut() = Vault::close(!k, v);
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_load_returns_some_if_key_matches(s: HashSize, k: Zobrist, v: Transposition) {
        let mut tt = TranspositionTable::new(s);
        *tt.entries[k].get_mut() = Vault::close(k, v);
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_stores_value_if_slot_is_empty(s: HashSize, k: Zobrist, mut v: Transposition) {
        let mut tt = TranspositionTable::new(s);
        tt[k] = zero();
        tt.store(k, v);
        v.age = *tt.age.get_mut();
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_store_replaces_value_if_key_does_not_match(
        s: HashSize,
        k: Zobrist,
        u: Transposition,
        mut v: Transposition,
    ) {
        let mut tt = TranspositionTable::new(s);
        *tt.entries[k].get_mut() = Vault::close(!k, u);
        tt.store(k, v);
        v.age = *tt.age.get_mut();
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_store_replaces_value_if_age_is_different(
        s: HashSize,
        k: Zobrist,
        #[filter(#u.age != zero())] u: Transposition,
        mut v: Transposition,
    ) {
        let mut tt = TranspositionTable::new(s);
        tt.store(k, u);
        tt.age();
        tt.store(k, v);
        v.age = *tt.age.get_mut();
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_store_replaces_value_if_deeper(
        s: HashSize,
        k: Zobrist,
        u: Transposition,
        #[filter(#v.depth >= #u.depth)] mut v: Transposition,
    ) {
        let mut tt = TranspositionTable::new(s);
        tt.store(k, u);
        tt.store(k, v);
        v.age = *tt.age.get_mut();
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_allocates_up_to_vt_size(s: HashSize) {
        assert!(vt_size(s) >= ValueTable::new(s).len() * size_of::<u32>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_resizes_up_to_vt_size(s: HashSize, t: HashSize) {
        let mut vt = ValueTable::new(s);
        vt.resize(t);
        assert!(vt_size(t) >= vt.len() * size_of::<u32>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_load_returns_none_if_slot_is_empty(s: HashSize, k: Zobrist) {
        let mut vt = ValueTable::new(s);
        *vt.entries[k].get_mut() = zero();
        assert_eq!(vt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_load_returns_none_if_key_does_not_match(s: HashSize, k: Zobrist, v: Value) {
        let mut vt = ValueTable::new(s);
        *vt.entries[k].get_mut() = Vault::close(!k, v);
        assert_eq!(vt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_load_returns_some_if_key_matches(s: HashSize, k: Zobrist, v: Value) {
        let mut vt = ValueTable::new(s);
        *vt.entries[k].get_mut() = Vault::close(k, v);
        assert_eq!(vt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_stores_value_if_slot_is_empty(s: HashSize, k: Zobrist, v: Value) {
        let mut vt = ValueTable::new(s);
        *vt.entries[k].get_mut() = zero();
        vt.store(k, v);
        assert_eq!(vt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_store_always_replaces_value_if_one_exists(
        s: HashSize,
        k: Zobrist,
        u: Value,
        l: Zobrist,
        v: Value,
    ) {
        let mut vt = ValueTable::new(s);
        *vt.entries[k].get_mut() = Vault::close(l, u);
        vt.store(k, v);
        assert_eq!(vt.load(k), Some(v));
    }
}

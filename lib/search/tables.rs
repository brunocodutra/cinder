use crate::chess::{Move, Zobrist};
use crate::nnue::Value;
use crate::search::{HashSize, ThreadCount, Transposition};
use crate::util::{Atomic, HugePages, Int, Memory, Prefetch, Vault};
use derive_more::with_trait::Debug;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::Ordering::Relaxed;
use std::{cell::UnsafeCell, mem::MaybeUninit};
use std::{ptr, slice};

#[derive(Debug)]
#[debug("TranspositionTable({})", entries.len())]
pub struct TranspositionTable {
    entries: HugePages<Atomic<Vault<Transposition, u64>>>,
}

impl TranspositionTable {
    #[inline(always)]
    const fn size_to_len(size: HashSize) -> usize {
        size.get() / size_of::<Vault<Transposition, u64>>()
    }

    #[inline(always)]
    pub fn new(size: HashSize) -> Self {
        Self {
            entries: HugePages::zeroed(Self::size_to_len(size).cast()),
        }
    }

    #[inline(always)]
    pub fn resize(&mut self, size: HashSize) {
        self.entries.zeroed_in_place(Self::size_to_len(size).cast());
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn store(&self, key: Zobrist, new: Transposition) {
        if !self.entries.is_empty() {
            let slot = &self.entries[key];
            let Some(old) = slot.load(Relaxed).open(key) else {
                return slot.store(Vault::close(key, new), Relaxed);
            };

            if new.best.is_some_and(|m| m.is_quiet()) || new.depth > old.depth {
                slot.store(Vault::close(key, new), Relaxed);
            }
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn load(&self, key: Zobrist) -> Option<Transposition> {
        if !self.entries.is_empty() {
            self.entries[key].load(Relaxed).open(key)
        } else {
            None
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn prefetch(&self, key: Zobrist) {
        if !self.entries.is_empty() {
            ptr::from_ref(&self.entries[key]).prefetch();
        }
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
    entries: HugePages<Atomic<Vault<Value, u64>>>,
}

impl ValueTable {
    #[inline(always)]
    const fn size_to_len(threads: ThreadCount) -> usize {
        (2 + threads.cast::<usize>().next_multiple_of(2)) << 17
    }

    #[inline(always)]
    pub fn new(threads: ThreadCount) -> Self {
        Self {
            entries: HugePages::zeroed(Self::size_to_len(threads).cast()),
        }
    }

    #[inline(always)]
    pub fn resize(&mut self, threads: ThreadCount) {
        self.entries
            .zeroed_in_place(Self::size_to_len(threads).cast());
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn store(&self, key: Zobrist, value: Value) {
        self.entries[key].store(Vault::close(key, value), Relaxed);
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn load(&self, key: Zobrist) -> Option<Value> {
        self.entries[key].load(Relaxed).open(key)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn prefetch(&self, key: Zobrist) {
        ptr::from_ref(&self.entries[key]).prefetch();
    }
}

impl const Deref for ValueTable {
    type Target = [UnsafeCell<MaybeUninit<u64>>];

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
    fn tt_allocates_up_to_hash_size(s: HashSize) {
        assert!(s >= TranspositionTable::new(s).len() * size_of::<Transposition>());
    }

    #[proptest]
    fn tt_resizes_up_to_hash_size(s: HashSize, t: HashSize) {
        let mut tt = TranspositionTable::new(s);
        tt.resize(t);
        assert!(t >= tt.len() * size_of::<Transposition>());
    }

    #[proptest]
    fn tt_load_does_nothing_if_hash_size_is_zero(k: Zobrist) {
        assert_eq!(TranspositionTable::new(HashSize::new(0)).load(k), None);
    }

    #[proptest]
    fn tt_load_returns_none_if_slot_is_empty(s: HashSize, k: Zobrist) {
        let mut tt = TranspositionTable::new(s);
        tt.entries[k] = zero();
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    fn tt_load_returns_none_if_key_does_not_match(s: HashSize, k: Zobrist, v: Transposition) {
        let mut tt = TranspositionTable::new(s);
        *tt.entries[k].get_mut() = Vault::close(!k, v);
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    fn tt_load_returns_some_if_key_matches(s: HashSize, k: Zobrist, v: Transposition) {
        let mut tt = TranspositionTable::new(s);
        *tt.entries[k].get_mut() = Vault::close(k, v);
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    fn tt_stores_nothing_if_hash_size_is_zero(k: Zobrist, v: Transposition) {
        TranspositionTable::new(HashSize::new(0)).store(k, v);
    }

    #[proptest]
    fn tt_stores_value_if_slot_is_empty(s: HashSize, k: Zobrist, v: Transposition) {
        let mut tt = TranspositionTable::new(s);
        tt[k] = zero();
        tt.store(k, v);
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    fn tt_store_always_replaces_value_if_one_exists(
        s: HashSize,
        k: Zobrist,
        u: Transposition,
        l: Zobrist,
        v: Transposition,
    ) {
        let mut tt = TranspositionTable::new(s);
        *tt.entries[k].get_mut() = Vault::close(l, v);
        tt.store(k, u);
        assert_eq!(tt.load(k), Some(u));
    }

    #[proptest]
    fn vt_load_returns_none_if_slot_is_empty(s: ThreadCount, k: Zobrist) {
        let mut vt = ValueTable::new(s);
        vt.entries[k] = zero();
        assert_eq!(vt.load(k), None);
    }

    #[proptest]
    fn vt_load_returns_none_if_key_does_not_match(s: ThreadCount, k: Zobrist, v: Value) {
        let mut vt = ValueTable::new(s);
        *vt.entries[k].get_mut() = Vault::close(!k, v);
        assert_eq!(vt.load(k), None);
    }

    #[proptest]
    fn vt_load_returns_some_if_key_matches(s: ThreadCount, k: Zobrist, v: Value) {
        let mut vt = ValueTable::new(s);
        *vt.entries[k].get_mut() = Vault::close(k, v);
        assert_eq!(vt.load(k), Some(v));
    }

    #[proptest]
    fn vt_stores_value_if_slot_is_empty(s: ThreadCount, k: Zobrist, v: Value) {
        let mut vt = ValueTable::new(s);
        vt[k] = zero();
        vt.store(k, v);
        assert_eq!(vt.load(k), Some(v));
    }

    #[proptest]
    fn vt_store_always_replaces_value_if_one_exists(
        s: ThreadCount,
        k: Zobrist,
        u: Value,
        l: Zobrist,
        v: Value,
    ) {
        let mut vt = ValueTable::new(s);
        *vt.entries[k].get_mut() = Vault::close(l, v);
        vt.store(k, u);
        assert_eq!(vt.load(k), Some(u));
    }
}

use crate::search::{HashSize, ThreadCount, Transposition};
use crate::util::{Atomic, HugePages, Int, Memory, Prefetch, Vault};
use crate::{chess::Zobrist, nnue::Value};
use derive_more::with_trait::{Debug, Deref, DerefMut};
use std::{ptr, sync::atomic::Ordering};

#[derive(Debug, Deref, DerefMut)]
#[debug("TranspositionTable({})", _0.len())]
pub struct TranspositionTable(HugePages<Atomic<Vault<Transposition, u64>>>);

impl TranspositionTable {
    #[inline(always)]
    const fn size_to_len(size: HashSize) -> usize {
        size.get() / size_of::<Vault<Transposition, u64>>()
    }

    #[inline(always)]
    pub fn new(size: HashSize) -> Self {
        Self(HugePages::zeroed(Self::size_to_len(size).cast()))
    }

    #[inline(always)]
    pub fn resize(&mut self, size: HashSize) {
        self.0.zeroed_in_place(Self::size_to_len(size).cast());
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn store(&self, zobrist: Zobrist, tpos: Transposition) {
        if !self.0.is_empty() {
            self.0[zobrist].store(Vault::close(zobrist, tpos), Ordering::Relaxed);
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn load(&self, zobrist: Zobrist) -> Option<Transposition> {
        if !self.0.is_empty() {
            self.0[zobrist].load(Ordering::Relaxed).open(zobrist)
        } else {
            None
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn prefetch(&self, zobrist: Zobrist) {
        if !self.0.is_empty() {
            ptr::from_ref(&self.0[zobrist]).prefetch();
        }
    }
}

#[derive(Debug, Deref, DerefMut)]
#[debug("ValueTable({})", _0.len())]
pub struct ValueTable(HugePages<Atomic<Vault<Value, u64>>>);

impl ValueTable {
    #[inline(always)]
    const fn size_to_len(threads: ThreadCount) -> usize {
        (2 + threads.cast::<usize>().next_multiple_of(2)) << 17
    }

    #[inline(always)]
    pub fn new(threads: ThreadCount) -> Self {
        Self(HugePages::zeroed(Self::size_to_len(threads).cast()))
    }

    #[inline(always)]
    pub fn resize(&mut self, threads: ThreadCount) {
        self.0.zeroed_in_place(Self::size_to_len(threads).cast());
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn store(&self, zobrist: Zobrist, value: Value) {
        self.0[zobrist].store(Vault::close(zobrist, value), Ordering::Relaxed);
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn load(&self, zobrist: Zobrist) -> Option<Value> {
        self.0[zobrist].load(Ordering::Relaxed).open(zobrist)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn prefetch(&self, zobrist: Zobrist) {
        ptr::from_ref(&self.0[zobrist]).prefetch();
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
    fn tt_allocates_up_to_hash_size(s: HashSize) {
        assert!(s >= TranspositionTable::new(s).len() * size_of::<Transposition>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_resizes_up_to_hash_size(s: HashSize, t: HashSize) {
        let mut tt = TranspositionTable::new(s);
        tt.resize(t);
        assert!(t >= tt.len() * size_of::<Transposition>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_load_does_nothing_if_hash_size_is_zero(k: Zobrist) {
        assert_eq!(TranspositionTable::new(HashSize::new(0)).load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_load_returns_none_if_slot_is_empty(s: HashSize, k: Zobrist) {
        let mut tt = TranspositionTable::new(s);
        tt[k] = zero();
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_load_returns_none_if_key_does_not_match(s: HashSize, k: Zobrist, v: Transposition) {
        let mut tt = TranspositionTable::new(s);
        *tt[k].get_mut() = Vault::close(!k, v);
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_load_returns_some_if_key_matches(s: HashSize, k: Zobrist, v: Transposition) {
        let mut tt = TranspositionTable::new(s);
        *tt[k].get_mut() = Vault::close(k, v);
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_stores_nothing_if_hash_size_is_zero(k: Zobrist, v: Transposition) {
        TranspositionTable::new(HashSize::new(0)).store(k, v);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_stores_value_if_slot_is_empty(s: HashSize, k: Zobrist, v: Transposition) {
        let mut tt = TranspositionTable::new(s);
        tt[k] = zero();
        tt.store(k, v);
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn tt_store_always_replaces_value_if_one_exists(
        s: HashSize,
        k: Zobrist,
        u: Transposition,
        l: Zobrist,
        v: Transposition,
    ) {
        let mut tt = TranspositionTable::new(s);
        *tt[k].get_mut() = Vault::close(l, v);
        tt.store(k, u);
        assert_eq!(tt.load(k), Some(u));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_load_returns_none_if_slot_is_empty(s: ThreadCount, k: Zobrist) {
        let mut tt = ValueTable::new(s);
        tt[k] = zero();
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_load_returns_none_if_key_does_not_match(s: ThreadCount, k: Zobrist, v: Value) {
        let mut tt = ValueTable::new(s);
        *tt[k].get_mut() = Vault::close(!k, v);
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_load_returns_some_if_key_matches(s: ThreadCount, k: Zobrist, v: Value) {
        let mut tt = ValueTable::new(s);
        *tt[k].get_mut() = Vault::close(k, v);
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_stores_value_if_slot_is_empty(s: ThreadCount, k: Zobrist, v: Value) {
        let mut tt = ValueTable::new(s);
        tt[k] = zero();
        tt.store(k, v);
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_store_always_replaces_value_if_one_exists(
        s: ThreadCount,
        k: Zobrist,
        u: Value,
        l: Zobrist,
        v: Value,
    ) {
        let mut tt = ValueTable::new(s);
        *tt[k].get_mut() = Vault::close(l, v);
        tt.store(k, u);
        assert_eq!(tt.load(k), Some(u));
    }
}

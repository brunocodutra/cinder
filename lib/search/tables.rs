use crate::chess::Zobrist;
use crate::search::{HashSize, Transposition, Value};
use crate::util::{Atomic, HugePages, Memory, Num, Prefetch, Vault};
use derive_more::with_trait::{Debug, Deref, DerefMut};
use std::{ops::Shr, ptr, sync::atomic::Ordering};

#[inline(always)]
const fn tt_size(size: HashSize) -> usize {
    size.get().max(1 << 18) - vt_size(size)
}

#[inline(always)]
const fn vt_size(size: HashSize) -> usize {
    size.get().max(1 << 18).shr(3) // ~12.5%
}

#[derive(Debug, Deref, DerefMut)]
#[debug("TranspositionTable({})", _0.len())]
pub struct TranspositionTable(HugePages<Atomic<Vault<Transposition, u64>>>);

impl TranspositionTable {
    #[inline(always)]
    pub fn new(size: HashSize) -> Self {
        Self(HugePages::zeroed(tt_size(size).shr(3u32).cast()))
    }

    #[inline(always)]
    pub fn resize(&mut self, size: HashSize) {
        self.0.zeroed_in_place(tt_size(size).shr(3u32).cast());
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn store(&self, zobrist: Zobrist, tpos: Transposition) {
        self.0[zobrist].store(Vault::close(zobrist, tpos), Ordering::Relaxed);
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn load(&self, zobrist: Zobrist) -> Option<Transposition> {
        self.0[zobrist].load(Ordering::Relaxed).open(zobrist)
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn prefetch(&self, zobrist: Zobrist) {
        ptr::from_ref(&self.0[zobrist]).prefetch();
    }
}

#[derive(Debug, Deref, DerefMut)]
#[debug("ValueTable({})", _0.len())]
pub struct ValueTable(HugePages<Atomic<Vault<Value, u64>>>);

impl ValueTable {
    #[inline(always)]
    pub fn new(size: HashSize) -> Self {
        Self(HugePages::zeroed(vt_size(size).shr(3u32).cast()))
    }

    #[inline(always)]
    pub fn resize(&mut self, size: HashSize) {
        self.0.zeroed_in_place(vt_size(size).shr(3u32).cast());
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
    fn vt_allocates_up_to_vt_size(s: HashSize) {
        assert!(vt_size(s) >= ValueTable::new(s).len() * size_of::<u64>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_resizes_up_to_vt_size(s: HashSize, t: HashSize) {
        let mut vt = ValueTable::new(s);
        vt.resize(t);
        assert!(vt_size(t) >= vt.len() * size_of::<u64>());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_load_returns_none_if_slot_is_empty(s: HashSize, k: Zobrist) {
        let mut tt = ValueTable::new(s);
        tt[k] = zero();
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_load_returns_none_if_key_does_not_match(s: HashSize, k: Zobrist, v: Value) {
        let mut tt = ValueTable::new(s);
        *tt[k].get_mut() = Vault::close(!k, v);
        assert_eq!(tt.load(k), None);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_load_returns_some_if_key_matches(s: HashSize, k: Zobrist, v: Value) {
        let mut tt = ValueTable::new(s);
        *tt[k].get_mut() = Vault::close(k, v);
        assert_eq!(tt.load(k), Some(v));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn vt_stores_value_if_slot_is_empty(s: HashSize, k: Zobrist, v: Value) {
        let mut tt = ValueTable::new(s);
        tt[k] = zero();
        tt.store(k, v);
        assert_eq!(tt.load(k), Some(v));
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
        let mut tt = ValueTable::new(s);
        *tt[k].get_mut() = Vault::close(l, v);
        tt.store(k, u);
        assert_eq!(tt.load(k), Some(u));
    }
}

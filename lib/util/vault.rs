use crate::util::*;
use bytemuck::{Pod, Zeroable};
use derive_more::with_trait::Debug;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

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
pub struct Vault<T: Binary, U: Unsigned> {
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

impl<T, U, R, const B: u32> Vault<T, U>
where
    T: Binary<Bits = Bits<R, B>>,
    U: Unsigned,
    R: Unsigned,
{
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
        const { assert!(B <= U::BITS && U::BITS <= <Key as Int>::Repr::BITS) }

        key.push(value.encode());

        Vault {
            bits: key.cast(),
            phantom: PhantomData,
        }
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn open(self, key: Key) -> Option<T> {
        const { assert!(B <= U::BITS && U::BITS <= <Key as Int>::Repr::BITS) }

        if self.matches(key) {
            Some(Binary::decode(self.bits.convert::<Key>().assume().pop()))
        } else {
            None
        }
    }

    /// Whether this key matches the lock.
    #[inline(always)]
    pub fn matches(&self, key: Key) -> bool {
        !self.is_empty() && (self.bits >> B.cast()) == key.slice(..(U::BITS - B)).cast()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use test_strategy::proptest;

    type MockVault = Vault<u8, u64>;

    #[proptest]
    fn opening_vault_with_correct_key_succeeds(k: Key, v: u8) {
        assert_eq!(MockVault::close(k, v).open(k), Some(v));
    }

    #[proptest]
    fn opening_vault_with_wrong_key_fails(k: Key, #[filter(#l != #k)] l: Key, v: u8) {
        assert_eq!(MockVault::close(k, v).open(l), None);
    }
}

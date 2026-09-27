use crate::util::*;
use bytemuck::{Pod, Zeroable, zeroed};
use derive_more::with_trait::Debug;
use std::marker::{Destruct, PhantomData};
use std::ops::{Index, IndexMut};

/// The key to a [`Vault`].
pub type Key = Bits<u64, 64>;

const impl<T> Index<Key> for [T] {
    type Output = T;

    #[inline(always)]
    fn index(&self, key: Key) -> &Self::Output {
        let idx = ((key.cast::<u128>() * self.len().cast::<u128>()) >> 64) as usize;
        self.get(idx).assume()
    }
}

const impl<T> IndexMut<Key> for [T] {
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

const impl<T: Binary, U: Unsigned> Clone for Vault<T, U> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

const impl<T, U, R, const B: u32> Vault<T, U>
where
    T: [const] Destruct + [const] Binary<Bits = Bits<R, B>>,
    U: [const] Unsigned,
    R: [const] Unsigned,
{
    // Whether the vault is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.bits == zeroed()
    }

    // Constructs an empty vault.
    #[inline(always)]
    pub fn empty() -> Self {
        Vault {
            bits: zeroed(),
            phantom: PhantomData,
        }
    }

    /// Locks `value` in the vault with `key`.
    #[inline(always)]
    #[expect(clippy::needless_pass_by_value)]
    pub fn close(mut key: Key, value: T) -> Self {
        const { assert!(B <= U::BITS && U::BITS <= <Key as Num>::Repr::BITS) }

        key.push(value.encode());

        Vault {
            bits: key.cast(),
            phantom: PhantomData,
        }
    }

    /// Returns the value stored in the vault if `key` matches.
    #[inline(always)]
    pub fn open(self, key: Key) -> Option<T> {
        const { assert!(B <= U::BITS && U::BITS <= <Key as Num>::Repr::BITS) }
        if self.matches(key) { self.peek() } else { None }
    }

    /// Returns the value stored in the vault, or `None` when it is empty.
    #[inline(always)]
    pub fn peek(self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(Binary::decode(self.bits.convert::<Key>().assume().pop()))
        }
    }

    /// Whether `key` can open this vault.
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

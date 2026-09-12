use crate::simd::*;
use crate::util::{Assume, Binary, Bits, Int, Num};
use bytemuck::{NoUninit, Pod, Zeroable, zeroed};
use derive_more::with_trait::{Debug, Deref, Display, IntoIterator};
use std::{iter::FusedIterator, ops::*};

/// A numeric identifier for a piece on the board, or none.
#[derive(Debug, Copy, Hash, Zeroable, NoUninit)]
#[derive_const(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Idx(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Idx as Num>::Repr);

const unsafe impl Num for Idx {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 15;
}

const unsafe impl Int for Idx {}

const impl Idx {
    pub const LEN: usize = Self::MAX as usize + 1;

    pub const KING: Self = zeroed();

    /// This index's [`IdxSet`].
    #[inline(always)]
    pub fn to_set(self) -> IdxSet {
        IdxSet::new(1 << self.0)
    }
}

const impl Binary for Idx {
    type Bits = Bits<u8, 4>;

    #[inline(always)]
    fn encode(&self) -> Self::Bits {
        self.convert().assume()
    }

    #[inline(always)]
    fn decode(bits: Self::Bits) -> Self {
        bits.convert().assume()
    }
}

const impl<T> Index<Idx> for [T; Idx::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, idx: Idx) -> &Self::Output {
        self.get(idx.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Idx> for [T; Idx::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, idx: Idx) -> &mut Self::Output {
        self.get_mut(idx.cast::<usize>()).assume()
    }
}

/// A set of [`Idx`]s.
#[derive(Debug, Display, Copy, Hash, Zeroable, Pod)]
#[derive_const(Default, Clone, PartialEq, Eq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("IdxSet({self})")]
#[display("{_0:016b}")]
#[repr(transparent)]
pub struct IdxSet(pub <IdxSet as Num>::Repr);

const unsafe impl Num for IdxSet {
    type Repr = u16;
    const MIN: Self::Repr = u16::MIN;
    const MAX: Self::Repr = u16::MAX;
}

const unsafe impl Int for IdxSet {}

const impl IdxSet {
    /// An empty set of [`Place`]s.
    #[inline(always)]
    pub fn empty() -> Self {
        zeroed()
    }

    /// The number of [`Place`]s in the set.
    #[inline(always)]
    pub fn len(self) -> usize {
        self.0.count_ones().cast::<usize>()
    }

    /// Whether the set is empty.
    #[inline(always)]
    pub fn is_empty(self) -> bool {
        self == Self::empty()
    }

    /// Whether this [`Idx`] is in the set.
    #[inline(always)]
    pub fn contains(self, idx: Idx) -> bool {
        self & idx.to_set() != zeroed()
    }

    /// An iterator over the [`Idx`]s in this set.
    #[inline(always)]
    pub fn iter(self) -> Indices {
        Indices(self)
    }
}

const impl Deref for IdxSet {
    type Target = u16;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const impl Not for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        Self(self.0.not())
    }
}

const impl BitAnd for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self::Output {
        self.bitand(rhs.0)
    }
}

const impl BitAnd<u16> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: u16) -> Self::Output {
        Self(self.0.bitand(rhs))
    }
}

impl<T: MaskElement> BitAnd<M<T, 16>> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: M<T, 16>) -> Self::Output {
        Self(self.0.bitand(rhs.to_bitmask() as u16))
    }
}

impl<T: MaskElement> BitAnd<Mask<T, 16>> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitand(self, rhs: Mask<T, 16>) -> Self::Output {
        Self(self.0.bitand(rhs.to_bitmask() as u16))
    }
}

const impl BitAndAssign for IdxSet {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) {
        self.bitand_assign(rhs.0);
    }
}

const impl BitAndAssign<u16> for IdxSet {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: u16) {
        self.0.bitand_assign(rhs);
    }
}

impl<T: MaskElement> BitAndAssign<M<T, 16>> for IdxSet {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: M<T, 16>) {
        self.0.bitand_assign(rhs.to_bitmask() as u16);
    }
}

impl<T: MaskElement> BitAndAssign<Mask<T, 16>> for IdxSet {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Mask<T, 16>) {
        self.0.bitand_assign(rhs.to_bitmask() as u16);
    }
}

const impl BitOr for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.bitor(rhs.0)
    }
}

const impl BitOr<u16> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: u16) -> Self::Output {
        Self(self.0.bitor(rhs))
    }
}

impl<T: MaskElement> BitOr<M<T, 16>> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: M<T, 16>) -> Self::Output {
        Self(self.0.bitor(rhs.to_bitmask() as u16))
    }
}

impl<T: MaskElement> BitOr<Mask<T, 16>> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitor(self, rhs: Mask<T, 16>) -> Self::Output {
        Self(self.0.bitor(rhs.to_bitmask() as u16))
    }
}

const impl BitOrAssign for IdxSet {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        self.bitor_assign(rhs.0);
    }
}

const impl BitOrAssign<u16> for IdxSet {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: u16) {
        self.0.bitor_assign(rhs);
    }
}

impl<T: MaskElement> BitOrAssign<M<T, 16>> for IdxSet {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: M<T, 16>) {
        self.0.bitor_assign(rhs.to_bitmask() as u16);
    }
}

impl<T: MaskElement> BitOrAssign<Mask<T, 16>> for IdxSet {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Mask<T, 16>) {
        self.0.bitor_assign(rhs.to_bitmask() as u16);
    }
}

const impl BitXor for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self::Output {
        self.bitxor(rhs.0)
    }
}

const impl BitXor<u16> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: u16) -> Self::Output {
        Self(self.0.bitxor(rhs))
    }
}

impl<T: MaskElement> BitXor<M<T, 16>> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: M<T, 16>) -> Self::Output {
        Self(self.0.bitxor(rhs.to_bitmask() as u16))
    }
}

impl<T: MaskElement> BitXor<Mask<T, 16>> for IdxSet {
    type Output = Self;

    #[inline(always)]
    fn bitxor(self, rhs: Mask<T, 16>) -> Self::Output {
        Self(self.0.bitxor(rhs.to_bitmask() as u16))
    }
}

const impl BitXorAssign for IdxSet {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) {
        self.bitxor_assign(rhs.0);
    }
}

const impl BitXorAssign<u16> for IdxSet {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: u16) {
        self.0.bitxor_assign(rhs);
    }
}

impl<T: MaskElement> BitXorAssign<M<T, 16>> for IdxSet {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: M<T, 16>) {
        self.0.bitxor_assign(rhs.to_bitmask() as u16);
    }
}

impl<T: MaskElement> BitXorAssign<Mask<T, 16>> for IdxSet {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Mask<T, 16>) {
        self.0.bitxor_assign(rhs.to_bitmask() as u16);
    }
}

impl<T: MaskElement> From<M<T, 16>> for IdxSet {
    #[inline(always)]
    fn from(mask: M<T, 16>) -> Self {
        IdxSet(mask.to_bitmask() as u16)
    }
}

#[cfg(target_feature = "avx512f")]
impl<T: MaskElement> From<IdxSet> for M<T, 16> {
    #[inline(always)]
    fn from(indices: IdxSet) -> Self {
        M::from_bitmask(indices.cast())
    }
}

const impl IntoIterator for IdxSet {
    type Item = Idx;
    type IntoIter = Indices;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        Indices(self)
    }
}

/// An iterator over the [`Idx`]s in an [`IdxSet`].
#[derive(Debug)]
pub struct Indices(IdxSet);

const impl Indices {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

const impl Iterator for Indices {
    type Item = Idx;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_empty() {
            None
        } else {
            let idx: Idx = self.0.trailing_zeros().convert().assume();
            self.0 ^= idx.to_set();
            Some(idx)
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for Indices {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }
}

impl FusedIterator for Indices {}

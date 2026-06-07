use crate::util::{Assume, Bounded, Int, Num};
use bytemuck::{NoUninit, Zeroable};
use std::ops::{Index, IndexMut};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, NoUninit)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PlyRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PlyRepr as Num>::Repr);

const unsafe impl Num for PlyRepr {
    type Repr = i16;

    const MIN: Self::Repr = 0;

    #[cfg(not(test))]
    const MAX: Self::Repr = 159;

    #[cfg(test)]
    const MAX: Self::Repr = 7;
}

const unsafe impl Int for PlyRepr {}

/// The number of half-moves played.
pub type Ply = Bounded<PlyRepr>;

const impl<T> Index<Ply> for [T; Ply::MAX as usize + 1] {
    type Output = T;

    #[inline(always)]
    fn index(&self, p: Ply) -> &Self::Output {
        self.get(p.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Ply> for [T; Ply::MAX as usize + 1] {
    #[inline(always)]
    fn index_mut(&mut self, p: Ply) -> &mut Self::Output {
        self.get_mut(p.cast::<usize>()).assume()
    }
}

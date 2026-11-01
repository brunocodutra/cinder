use crate::util::{Bounded, Int, Num};
use bytemuck::{NoUninit, Zeroable};
use std::ops::{Index, IndexMut};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Zeroable, NoUninit)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct PlyRepr(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <PlyRepr as Num>::Repr);

const unsafe impl Num for PlyRepr {
    type Repr = i16;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = (Ply::LEN - 1).cast();
}

const unsafe impl Int for PlyRepr {}

/// The number of half-moves played.
pub type Ply = Bounded<PlyRepr>;

const impl Ply {
    #[cfg(not(test))]
    pub const LEN: usize = 160;

    #[cfg(test)]
    pub const LEN: usize = 8;
}

const impl<T> Index<Ply> for [T; Ply::LEN] {
    type Output = T;

    #[inline(always)]
    fn index(&self, p: Ply) -> &Self::Output {
        unsafe { self.get_unchecked(p.cast::<usize>()) }
    }
}

const impl<T> IndexMut<Ply> for [T; Ply::LEN] {
    #[inline(always)]
    fn index_mut(&mut self, p: Ply) -> &mut Self::Output {
        unsafe { self.get_unchecked_mut(p.cast::<usize>()) }
    }
}

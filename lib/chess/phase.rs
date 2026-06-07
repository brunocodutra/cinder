use crate::util::{Assume, Int, Num};
use std::ops::{Index, IndexMut};

/// The game phase.
#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[repr(transparent)]
pub struct Phase(#[cfg_attr(test, strategy(Self::MIN..=Self::MAX))] <Phase as Num>::Repr);

const impl Phase {
    pub const LEN: usize = Phase::MAX as usize + 1;
}

const unsafe impl Num for Phase {
    type Repr = u8;
    const MIN: Self::Repr = 0;
    const MAX: Self::Repr = 7;
}

const unsafe impl Int for Phase {}

const impl<T> Index<Phase> for [T; Phase::MAX as usize + 1] {
    type Output = T;

    #[inline(always)]
    fn index(&self, p: Phase) -> &Self::Output {
        self.get(p.cast::<usize>()).assume()
    }
}

const impl<T> IndexMut<Phase> for [T; Phase::MAX as usize + 1] {
    #[inline(always)]
    fn index_mut(&mut self, p: Phase) -> &mut Self::Output {
        self.get_mut(p.cast::<usize>()).assume()
    }
}

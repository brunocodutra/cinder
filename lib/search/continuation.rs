use crate::chess::{Move, Position, Role};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::Assume;
use derive_more::with_trait::Debug;
use std::mem::MaybeUninit;

#[derive(Debug)]
pub struct Reply([[Graviton; 64]; 6]);

impl Default for Reply {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

impl Reply {
    #[inline(always)]
    fn graviton(&self, pos: &Position, m: Move) -> &Graviton {
        let piece = pos[m.whence()].assume().role() as usize;
        &self.0[piece][m.whither() as usize]
    }
}

impl Statistics for Reply {
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, m: Move) -> <Self::Stat as Stat>::Value {
        self.graviton(pos, m).get()
    }

    #[inline(always)]
    fn update(&self, pos: &Position, m: Move, delta: <Self::Stat as Stat>::Value) {
        self.graviton(pos, m).update(delta);
    }
}

#[derive(Debug)]
#[debug("Continuation")]
pub struct Continuation(Box<[[[Reply; 6]; 64]; 12]>);

impl Default for Continuation {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { Box::new_zeroed().assume_init() })
    }
}

impl Continuation {
    #[inline(always)]
    pub fn reply(&self, pos: &Position, m: Move) -> &Reply {
        let piece = pos[m.whence()].assume() as usize;
        let victim = pos[m.whither()].map_or(Role::King, |p| p.role()) as usize;
        &self.0[piece][m.whither() as usize][victim]
    }
}

use crate::chess::{Move, Position};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::Assume;
use derive_more::with_trait::Debug;
use std::mem::MaybeUninit;

#[derive(Debug)]
pub struct Reply([[[Graviton; 64]; 2]; 6]);

impl Default for Reply {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

impl Reply {
    #[inline(always)]
    fn graviton(&self, pos: &Position, m: Move) -> &Graviton {
        let role = pos.role_on(m.whence()).assume() as usize;
        &self.0[role][m.is_quiet() as usize][m.whither() as usize]
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
pub struct Continuation([[[Reply; 64]; 2]; 12]);

impl Default for Continuation {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

impl Continuation {
    #[inline(always)]
    pub fn reply(&self, pos: &Position, m: Move) -> &Reply {
        let piece = pos.piece_on(m.whence()).assume() as usize;
        &self.0[piece][m.is_quiet() as usize][m.whither() as usize]
    }
}

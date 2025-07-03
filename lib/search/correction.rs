use crate::chess::{Position, Zobrist};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::Integer;
use derive_more::with_trait::Debug;

const BUCKETS: usize = 8192;

/// Historical statistics about a [`Move`].
#[derive(Debug)]
#[debug("Correction")]
pub struct Correction([[<Self as Statistics<Zobrist>>::Stat; BUCKETS]; 2]);

impl Default for Correction {
    #[inline(always)]
    fn default() -> Self {
        Self([[Default::default(); BUCKETS]; 2])
    }
}

impl Correction {
    pub const LIMIT: i16 = 512;

    #[inline(always)]
    fn graviton(
        &mut self,
        pos: &Position,
        key: Zobrist,
    ) -> &mut <Self as Statistics<Zobrist>>::Stat {
        const { assert!(BUCKETS.is_power_of_two()) }
        &mut self.0[pos.turn() as usize][key.slice(..BUCKETS.trailing_zeros()).cast::<usize>()]
    }
}

impl Statistics<Zobrist> for Correction {
    type Stat = Graviton<{ -Self::LIMIT }, { Self::LIMIT }>;

    #[inline(always)]
    fn get(&mut self, pos: &Position, key: Zobrist) -> <Self::Stat as Stat>::Value {
        self.graviton(pos, key).get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, key: Zobrist, delta: <Self::Stat as Stat>::Value) {
        self.graviton(pos, key).update(delta);
    }
}

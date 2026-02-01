use crate::chess::Position;
use crate::search::{Graviton, Stat, Statistics};
use crate::util::{Bits, Int, Unsigned};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// Historical statistics about a [`Move`](`crate::chess::Move`).
#[derive(Debug, Zeroable)]
#[debug("Correction")]
pub struct Correction<const N: usize>([[Graviton; N]; 2]);

impl<const N: usize> const Default for Correction<N> {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl<U, const B: u32, const N: usize> const Statistics<Bits<U, B>> for Correction<N>
where
    U: [const] Unsigned,
{
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, key: Bits<U, B>) -> <Self::Stat as Stat>::Value {
        const { assert!(N.is_power_of_two()) }
        const { assert!(N.trailing_zeros() <= B) }
        self.0[pos.turn() as usize][key.slice(..N.trailing_zeros()).cast::<usize>()].get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, key: Bits<U, B>, delta: <Self::Stat as Stat>::Value) {
        const { assert!(N.is_power_of_two()) }
        const { assert!(N.trailing_zeros() <= B) }
        self.0[pos.turn() as usize][key.slice(..N.trailing_zeros()).cast::<usize>()].update(delta);
    }
}

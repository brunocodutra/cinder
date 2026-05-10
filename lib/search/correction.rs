use crate::chess::{Butterfly, Move, Position};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::{Binary, Bits, Num, Unsigned};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;
use std::marker::Destruct;

/// Learned corrections to evaluation relative to position structure.
#[derive(Debug, Zeroable)]
#[debug("Correction")]
pub struct Correction<const N: usize>([[Graviton; N]; 2]);

impl<const N: usize> const Default for Correction<N> {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl<T, U, const B: u32, const N: usize> const Statistics<T> for Correction<N>
where
    T: [const] Destruct + [const] Binary<Bits = Bits<U, B>>,
    U: [const] Unsigned,
{
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, key: T) -> <Self::Stat as Stat>::Value {
        const { assert!(N > 0) }
        const { assert!(N.is_power_of_two()) }
        const { assert!(N.trailing_zeros() <= B) }
        let idx = key.encode().slice(..N.trailing_zeros()).cast::<usize>();
        self.0[pos.turn() as usize][idx].get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, key: T, delta: <Self::Stat as Stat>::Value) {
        const { assert!(N > 0) }
        const { assert!(N.is_power_of_two()) }
        const { assert!(N.trailing_zeros() <= B) }
        let idx = key.encode().slice(..N.trailing_zeros()).cast::<usize>();
        self.0[pos.turn() as usize][idx].update(delta);
    }
}

/// Learned corrections to evaluation relative to historical [`Move`]s.
#[derive(Debug, Zeroable)]
#[debug("ContinuationCorrection")]
pub struct ContinuationCorrection([[Butterfly<[[Correction<1>; 2]; 2]>; 2]; 2]);

impl const Default for ContinuationCorrection {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

const impl ContinuationCorrection {
    #[inline(always)]
    pub fn get(&mut self, pos: &Position, m: Move) -> &mut Correction<1> {
        let (wc, wt) = (m.whence(), m.whither());
        let threats = [pos.threats().contains(wc), pos.threats().contains(wt)];
        &mut self.0[pos.turn() as usize][pos.is_check() as usize][wc as usize][wt as usize]
            [threats[0] as usize][threats[1] as usize]
    }
}

use crate::chess::{Butterfly, Move, Position};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::{Bits, Num, Unsigned};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

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

impl<U, const B: u32, const N: usize> const Statistics<Bits<U, B>> for Correction<N>
where
    U: [const] Unsigned,
{
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, key: Bits<U, B>) -> <Self::Stat as Stat>::Value {
        const { assert!(N > 0) }
        const { assert!(N.is_power_of_two()) }
        const { assert!(N.trailing_zeros() <= B) }
        self.0[pos.turn() as usize][key.slice(..N.trailing_zeros()).cast::<usize>()].get()
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, key: Bits<U, B>, delta: <Self::Stat as Stat>::Value) {
        const { assert!(N > 0) }
        const { assert!(N.is_power_of_two()) }
        const { assert!(N.trailing_zeros() <= B) }
        self.0[pos.turn() as usize][key.slice(..N.trailing_zeros()).cast::<usize>()].update(delta);
    }
}

impl const Statistics<()> for Correction<1> {
    type Stat = Graviton;

    #[inline(always)]
    fn get(&self, pos: &Position, (): ()) -> <Self::Stat as Stat>::Value {
        self.get(pos, Bits::<u8, 0>::default())
    }

    #[inline(always)]
    fn update(&mut self, pos: &Position, (): (), delta: <Self::Stat as Stat>::Value) {
        self.update(pos, Bits::<u8, 0>::default(), delta);
    }
}

/// Learned corrections to evaluation relative to historical [`Move`]s.
#[derive(Debug, Zeroable)]
#[debug("HistoryCorrection")]
pub struct HistoryCorrection([Butterfly<[[Correction<1>; 2]; 2]>; 2]);

impl const Default for HistoryCorrection {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl HistoryCorrection {
    #[inline(always)]
    pub const fn get(&mut self, pos: &Position, m: Move) -> &mut Correction<1> {
        let (wc, wt) = (m.whence(), m.whither());
        let threats = [pos.threats().contains(wc), pos.threats().contains(wt)];
        &mut self.0[pos.turn() as usize][wc as usize][wt as usize][threats[0] as usize]
            [threats[1] as usize]
    }
}

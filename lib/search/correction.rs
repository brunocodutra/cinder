use crate::chess::{Butterfly, Color, Move, Position};
use crate::search::{Graviton, Stat, Statistics};
use crate::util::{Binary, Bits, Num, Unsigned};
use bytemuck::{Zeroable, zeroed};
use derive_more::with_trait::Debug;

/// Learned corrections to evaluation relative to position structure.
#[derive(Debug, Zeroable)]
#[debug("Correction")]
pub struct Correction<const N: usize>([[Graviton; N]; 2]);

impl<const N: usize> Default for Correction<N> {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

/// Context about a [`Position`] for the purpose of correcting evaluations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CorrectionContext<T> {
    pub turn: Color,
    pub key: T,
}

impl<T> CorrectionContext<T> {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn new(pos: &Position, key: T) -> Self {
        CorrectionContext {
            turn: pos.turn(),
            key,
        }
    }
}

impl<T, U, const B: u32, const N: usize> Statistics<CorrectionContext<T>> for Correction<N>
where
    T: Binary<Bits = Bits<U, B>>,
    U: Unsigned,
{
    type Stat = Graviton;

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn get(&self, ctx: CorrectionContext<T>) -> <Self::Stat as Stat>::Value {
        const { assert!(N > 0) }
        const { assert!(N.is_power_of_two()) }
        const { assert!(N.trailing_zeros() <= B) }
        let idx = ctx.key.encode().slice(..N.trailing_zeros()).cast::<usize>();
        self.0[ctx.turn][idx].get()
    }

    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    fn update(&mut self, ctx: CorrectionContext<T>, delta: <Self::Stat as Stat>::Value) {
        const { assert!(N > 0) }
        const { assert!(N.is_power_of_two()) }
        const { assert!(N.trailing_zeros() <= B) }
        let idx = ctx.key.encode().slice(..N.trailing_zeros()).cast::<usize>();
        self.0[ctx.turn][idx].update(delta);
    }
}

/// Learned corrections to evaluation relative to historical [`Move`]s.
#[derive(Debug, Zeroable)]
#[debug("ContinuationCorrection")]
pub struct ContinuationCorrection([[Butterfly<[[Correction<1>; 2]; 2]>; 2]; 2]);

impl Default for ContinuationCorrection {
    #[inline(always)]
    fn default() -> Self {
        zeroed()
    }
}

impl ContinuationCorrection {
    #[inline(always)]
    #[cfg_attr(feature = "no_panic", no_panic::no_panic)]
    pub fn get(&mut self, pos: &Position, m: Move) -> &mut Correction<1> {
        let is_check = pos.is_check() as usize;
        let (wc, wt) = (m.whence(), m.whither());
        let attacked = [!pos.attackers(wc).is_empty(), !pos.attackers(wt).is_empty()];
        &mut self.0[pos.turn()][is_check][wc][wt][attacked[0] as usize][attacked[1] as usize]
    }
}

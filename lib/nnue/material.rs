use crate::chess::{Color, Perspective};
use crate::nnue::{Accumulator, Feature, Nnue};
use crate::util::{AlignTo64, Assume};
use derive_more::with_trait::Debug;

/// An accumulator for the psqt transformer.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Positional")]
pub struct Material(
    #[cfg_attr(test, map(|vs: [[i8; Self::LEN]; 2]| AlignTo64(vs.map(|v| v.map(i32::from)))))]
    AlignTo64<[[i32; Self::LEN]; 2]>,
);

impl Default for Material {
    #[inline(always)]
    fn default() -> Self {
        Material(AlignTo64([Nnue::psqt().fresh(); 2]))
    }
}

impl Accumulator for Material {
    const LEN: usize = 8;

    #[inline(always)]
    fn refresh(&mut self, side: Color) {
        self.0[side as usize] = Nnue::psqt().fresh();
    }

    #[inline(always)]
    fn add(&mut self, side: Color, feature: Feature) {
        Nnue::psqt().add(feature, &mut self.0[side as usize]);
    }

    #[inline(always)]
    fn remove(&mut self, side: Color, feature: Feature) {
        Nnue::psqt().remove(feature, &mut self.0[side as usize]);
    }

    #[inline(always)]
    fn replace(&mut self, side: Color, remove: Feature, add: Feature) {
        Nnue::psqt().replace(remove, add, &mut self.0[side as usize]);
    }

    #[inline(always)]
    fn evaluate(&self, turn: Color, phase: usize) -> i32 {
        self.0[turn as usize].get(phase).assume() - self.0[turn.flip() as usize].get(phase).assume()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::Feature;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn material_evaluation_is_antisymmetric(a: Material, c: Color, #[strategy(..8usize)] p: usize) {
        assert_eq!(a.evaluate(c, p), -a.evaluate(!c, p));
    }

    #[proptest]
    fn remove_reverses_add(a: Material, c: Color, f: Feature) {
        let mut b = a.clone();
        b.add(c, f);
        b.remove(c, f);
        assert_eq!(a, b);
    }

    #[proptest]
    fn replace_reverses_itself(a: Material, c: Color, x: Feature, y: Feature) {
        let mut b = a.clone();
        b.replace(c, x, y);
        b.replace(c, y, x);
        assert_eq!(a, b);
    }
}

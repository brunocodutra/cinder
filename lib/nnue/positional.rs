use crate::chess::{Color, Perspective};
use crate::nnue::{Accumulator, Feature, Nnue};
use crate::util::AlignTo64;
use derive_more::with_trait::Debug;

/// An accumulator for the feature transformer.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Positional")]
pub struct Positional(
    #[cfg_attr(test, map(|vs: [[i8; Self::LEN]; 2]| AlignTo64(vs.map(|v| v.map(i16::from)))))]
    AlignTo64<[[i16; Self::LEN]; 2]>,
);

impl Default for Positional {
    #[inline(always)]
    fn default() -> Self {
        Positional(AlignTo64([Nnue::ft().fresh(); 2]))
    }
}

impl Accumulator for Positional {
    const LEN: usize = 768;

    #[inline(always)]
    fn refresh(&mut self, side: Color) {
        self.0[side as usize] = Nnue::ft().fresh();
    }

    #[inline(always)]
    fn add(&mut self, side: Color, feature: Feature) {
        Nnue::ft().add(feature, &mut self.0[side as usize]);
    }

    #[inline(always)]
    fn remove(&mut self, side: Color, feature: Feature) {
        Nnue::ft().remove(feature, &mut self.0[side as usize]);
    }

    #[inline(always)]
    fn replace(&mut self, side: Color, remove: Feature, add: Feature) {
        Nnue::ft().replace(remove, add, &mut self.0[side as usize]);
    }

    #[inline(always)]
    fn evaluate(&self, turn: Color, phase: usize) -> i32 {
        Nnue::hidden(phase).forward(&self.0[turn as usize], &self.0[turn.flip() as usize]) << 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::Feature;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn remove_reverses_add(a: Positional, c: Color, f: Feature) {
        let mut b = a.clone();
        b.add(c, f);
        b.remove(c, f);
        assert_eq!(a, b);
    }

    #[proptest]
    fn replace_reverses_itself(a: Positional, c: Color, x: Feature, y: Feature) {
        let mut b = a.clone();
        b.replace(c, x, y);
        b.replace(c, y, x);
        assert_eq!(a, b);
    }
}

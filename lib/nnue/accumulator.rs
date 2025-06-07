use crate::chess::{Color, Flip};
use crate::nnue::{Feature, Nnue};
use crate::util::{AlignTo64, Assume};
use derive_more::with_trait::Debug;
use std::hint::unreachable_unchecked;

/// The NNUE accumulator.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[debug("Accumulator")]
pub struct Accumulator {
    #[cfg_attr(test, map(|vs: [[i8; Self::MATERIAL]; 2]| AlignTo64(vs.map(|v| v.map(i32::from)))))]
    material: AlignTo64<[[i32; Self::MATERIAL]; 2]>,
    #[cfg_attr(test, map(|vs: [[i8; Self::POSITIONAL]; 2]| AlignTo64(vs.map(|v| v.map(i16::from)))))]
    positional: AlignTo64<[[i16; Self::POSITIONAL]; 2]>,
}

impl Default for Accumulator {
    #[inline(always)]
    fn default() -> Self {
        Accumulator {
            material: AlignTo64([Nnue::material().fresh(); 2]),
            positional: AlignTo64([Nnue::positional().fresh(); 2]),
        }
    }
}

impl Accumulator {
    pub const MATERIAL: usize = 8;
    pub const POSITIONAL: usize = 768;

    #[inline(always)]
    pub fn refresh(&mut self, side: Color) {
        self.material[side as usize] = Nnue::material().fresh();
        self.positional[side as usize] = Nnue::positional().fresh();
    }

    #[inline(always)]
    pub fn update(&mut self, side: Color, sub: [Option<Feature>; 2], add: [Option<Feature>; 2]) {
        let material = Nnue::material();
        let positional = Nnue::positional();

        match (sub, add) {
            ([None, None], [Some(a1), None]) => {
                material.add(a1, &mut self.material[side as usize]);
                positional.add(a1, &mut self.positional[side as usize]);
            }

            ([Some(s1), None], [Some(a1), None]) => {
                material.sub_add(s1, a1, &mut self.material[side as usize]);
                positional.sub_add(s1, a1, &mut self.positional[side as usize]);
            }

            ([Some(s1), Some(s2)], [Some(a1), None]) => {
                material.sub_sub_add(s1, s2, a1, &mut self.material[side as usize]);
                positional.sub_sub_add(s1, s2, a1, &mut self.positional[side as usize]);
            }

            ([Some(s1), Some(s2)], [Some(a1), Some(a2)]) => {
                material.sub_sub_add_add(s1, s2, a1, a2, &mut self.material[side as usize]);
                positional.sub_sub_add_add(s1, s2, a1, a2, &mut self.positional[side as usize]);
            }

            _ => unsafe { unreachable_unchecked() },
        }
    }

    #[inline(always)]
    pub fn evaluate(&self, turn: Color, phase: usize) -> i32 {
        (phase < Self::MATERIAL).assume();

        let us = turn as usize;
        let them = turn.flip() as usize;
        let material = self.material[us][phase] - self.material[them][phase];
        let positional = Nnue::hidden(phase).forward(&self.positional[us], &self.positional[them]);
        material + 2 * positional
    }
}

use crate::{chess::Phase, util::Integer};
use bytemuck::{Zeroable, zeroed};
use std::slice;

mod accumulator;
mod evaluator;
mod feature;
mod output;
mod transformer;
mod value;

pub use accumulator::*;
pub use evaluator::*;
pub use feature::*;
pub use output::*;
pub use transformer::*;
pub use value::*;

const NNUE: Nnue = Nnue::new();

/// An Efficiently Updatable Neural Network.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
pub struct Nnue {
    transformer: Affine<i16, { Accumulator::LEN }>,
    output: [Output<{ Accumulator::LEN }>; Phase::LEN],
}

const unsafe fn copy_from_slice<T>(dst: &mut T, src: &[u8]) -> usize {
    let len = size_of_val(dst);
    let dst = unsafe { slice::from_raw_parts_mut(dst as *mut T as *mut u8, len) };
    dst.copy_from_slice(&src[..len]);
    len
}

impl Nnue {
    const fn new() -> Self {
        let bytes = include_bytes!("nnue/nnue.bin");
        let mut nnue: Self = zeroed();
        let mut cursor = 0;

        cursor += unsafe { copy_from_slice(&mut nnue.transformer.bias, &bytes[cursor..]) };
        cursor += unsafe { copy_from_slice(&mut nnue.transformer.weight, &bytes[cursor..]) };

        let mut phase = 0;
        while phase < Phase::LEN {
            cursor += unsafe { copy_from_slice(&mut nnue.output[phase].bias, &bytes[cursor..]) };
            phase += 1;
        }

        let mut phase = 0;
        while phase < Phase::LEN {
            cursor += unsafe { copy_from_slice(&mut nnue.output[phase].weight, &bytes[cursor..]) };
            phase += 1;
        }

        nnue
    }

    #[inline(always)]
    pub fn transformer() -> &'static Affine<i16, { Accumulator::LEN }> {
        &NNUE.transformer
    }

    #[inline(always)]
    pub fn output(phase: Phase) -> &'static Output<{ Accumulator::LEN }> {
        let idx = phase.cast::<usize>();
        unsafe { NNUE.output.get_unchecked(idx) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayVec;

    #[test]
    fn feature_transformer_does_not_overflow() {
        (0..Accumulator::LEN).for_each(|i| {
            let bias = Nnue::transformer().bias[i] as i32;
            let mut features = ArrayVec::<_, { Feature::LEN }>::from_iter(
                Nnue::transformer().weight.iter().map(|a| a[i] as i32),
            );

            for weights in features.as_chunks_mut::<768>().0 {
                let (small, _, _) = weights.select_nth_unstable(32);
                assert!(small.iter().fold(bias, |s, &v| s + v).abs() <= i16::MAX as i32);
                let (_, _, large) = weights.select_nth_unstable(735);
                assert!(large.iter().fold(bias, |s, &v| s + v).abs() <= i16::MAX as i32);
            }
        });
    }
}

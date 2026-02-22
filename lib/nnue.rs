use crate::chess::Phase;
use crate::util::{Int, Num};
use bytemuck::{Zeroable, zeroed};
use std::{ptr, slice};

mod accumulator;
mod evaluator;
mod feature;
mod hidden;
mod input;
mod layer;
mod output;
mod residual;
mod synapse;
mod transformer;

pub use accumulator::*;
pub use evaluator::*;
pub use feature::*;
pub use hidden::*;
pub use input::*;
pub use layer::*;
pub use output::*;
pub use residual::*;
pub use synapse::*;
pub use transformer::*;

/// Quantization constant for the feature transformer.
pub const FTQ: i16 = 255;

/// Quantization constant for the hidden layers.
pub const HLQ: i16 = 127;

/// Quantization scale for the hidden layers.
pub const HLS: i16 = 64;

const unsafe fn copy_bytes<T>(dst: &mut T, src: &[u8]) -> usize {
    let len = size_of_val(dst);
    let dst = unsafe { slice::from_raw_parts_mut(ptr::from_mut(dst).cast(), len) };
    dst.copy_from_slice(&src[..len]);
    len
}

const fn arrange_in_blocks<
    T: Copy,
    const I: usize,
    const O: usize,
    const B: usize,
    const N: usize,
>(
    input: &[[T; I]; O],
    output: &mut [[T; B]; N],
) {
    const { assert!(I.is_multiple_of(B)) }
    const { assert!(N == I * O / B) }

    let mut block = 0;
    while block < I / B {
        let mut i = 0;
        while i < O {
            let src = block * B;
            let dst = block * O + i;
            output[dst].copy_from_slice(&input[i][src..src + B]);
            i += 1;
        }

        block += 1;
    }
}

const fn interleave<T: Copy, const N: usize>(input: &[T; N], output: &mut [T; N], n: usize) {
    let mut i = 0;
    while i < N / 2 {
        let k = i / n;

        let mut j = 0;
        while j < n {
            output[k * 2 * n + j] = input[i + j];
            j += 1;
        }

        let mut j = 0;
        while j < n {
            output[k * 2 * n + n + j] = input[N / 2 + i + j];
            j += 1;
        }

        i += n;
    }
}

const NNUE: Nnue = Nnue::new();

/// An Efficiently Updatable Neural Network.
#[derive(Debug, Zeroable)]
pub struct Nnue {
    transformer: Transformer,
    nn: [Input<Residual<Hidden<Hidden<Output>>>>; Phase::LEN],
}

const impl Nnue {
    fn new() -> Self {
        let bytes = include_bytes!("nnue/nnue.bin");
        let mut nnue: Self = zeroed();
        let mut cursor = 0;

        cursor += unsafe { copy_bytes(&mut nnue.transformer.weight.0, &bytes[cursor..]) };
        cursor += unsafe { copy_bytes(&mut nnue.transformer.bias.0, &bytes[cursor..]) };

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase.cast::<usize>()];
            let mut weight = [[0i8; L1::LEN]; Ln::LEN / 2];
            cursor += unsafe { copy_bytes(&mut weight, &bytes[cursor..]) };
            let mut blocks = [[0i8; 4]; L1::LEN * Ln::LEN / 8];
            arrange_in_blocks(&weight, &mut blocks);
            interleave(&blocks, &mut nn.weight.0, Ln::LEN);
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase.cast::<usize>()];
            cursor += unsafe { copy_bytes(&mut nn.bias.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase.cast::<usize>()].next;
            cursor += unsafe { copy_bytes(&mut nn.weight.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase.cast::<usize>()].next.next;
            let mut weight = [[0f32; Ln::LEN]; Ln::LEN / 2];
            cursor += unsafe { copy_bytes(&mut weight, &bytes[cursor..]) };
            arrange_in_blocks(&weight, &mut nn.weight.0);
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase.cast::<usize>()].next.next;
            cursor += unsafe { copy_bytes(&mut nn.bias.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase.cast::<usize>()].next.next.next;
            let mut weight = [[0f32; Ln::LEN]; Ln::LEN / 2];
            cursor += unsafe { copy_bytes(&mut weight, &bytes[cursor..]) };
            arrange_in_blocks(&weight, &mut nn.weight.0);
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase.cast::<usize>()].next.next.next;
            cursor += unsafe { copy_bytes(&mut nn.bias.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase.cast::<usize>()].next.next.next.next;
            cursor += unsafe { copy_bytes(&mut nn.weight.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase.cast::<usize>()].next.next.next.next;
            let mut bias = 0f32;
            cursor += unsafe { copy_bytes(&mut bias, &bytes[cursor..]) };
            nn.bias.0 = [bias / nn.bias.0.len() as f32; _];
        }

        nnue
    }

    #[inline(always)]
    pub fn transformer() -> &'static Transformer {
        &NNUE.transformer
    }

    #[inline(always)]
    pub fn nn(phase: Phase) -> &'static Input<Residual<Hidden<Hidden<Output>>>> {
        let idx = phase.cast::<usize>();
        unsafe { NNUE.nn.get_unchecked(idx) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn feature_transformer_does_not_overflow() {
        (0..Accumulator::LEN).for_each(|i| {
            let transformer = Nnue::transformer();
            let bias = transformer.bias[i] as i32;
            let mut features = Vec::from_iter(transformer.weight.iter().map(|a| a[i] as i32));

            for weights in features.as_chunks_mut::<768>().0 {
                let (small, _, _) = weights.select_nth_unstable(32);
                assert!(small.iter().fold(bias, |s, &v| s + v).abs() <= i16::MAX as i32);
                let (_, _, large) = weights.select_nth_unstable(735);
                assert!(large.iter().fold(bias, |s, &v| s + v).abs() <= i16::MAX as i32);
            }
        });
    }
}

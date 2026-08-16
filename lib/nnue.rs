use crate::{chess::Phase, util::Int};
use bytemuck::{Zeroable, zeroed};
use std::{ptr, slice};

mod accumulator;
mod evaluator;
mod feature;
mod layer;
mod lin;
mod lnn;
mod lno;
mod lro;
mod synapse;
mod transformer;

pub use accumulator::*;
pub use evaluator::*;
pub use feature::*;
pub use layer::*;
pub use lin::*;
pub use lnn::*;
pub use lno::*;
pub use lro::*;
pub use synapse::*;
pub use transformer::*;

/// Quantization scale for the feature transformer.
pub const FTQ: i16 = 127;

/// Quantization scale for the hidden layers.
pub const HLQ: i16 = 75;

/// Conversion factor from quantized to floating point.
pub const I2F: f32 = (1 << 7) as f32 / (FTQ as f32 * FTQ as f32 * HLQ as f32);

/// Eval scale.
pub const F2V: f32 = 75.0;

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

static NNUE: Nnue = Nnue::new();

/// An Efficiently Updatable Neural Network.
#[derive(Debug, Zeroable)]
pub struct Nnue {
    transformer: Transformer,
    nn: [Lin<Lro<Lnn<Lnn<Lno>>>>; Phase::LEN],
}

const impl Nnue {
    fn new() -> Self {
        let bytes = include_bytes!(concat!(env!("OUT_DIR"), "/nnue.bin"));
        let mut nnue: Self = zeroed();
        let mut cursor = 0;

        cursor += unsafe { copy_bytes(&mut nnue.transformer.pp.0, &bytes[cursor..]) };
        cursor += unsafe { copy_bytes(&mut nnue.transformer.ti.0, &bytes[cursor..]) };
        cursor += unsafe { copy_bytes(&mut nnue.transformer.ka.0, &bytes[cursor..]) };
        cursor += unsafe { copy_bytes(&mut nnue.transformer.bias.0, &bytes[cursor..]) };

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase];
            let mut weight = [[0i8; Li::LEN]; Ln::LEN / 2];
            cursor += unsafe { copy_bytes(&mut weight, &bytes[cursor..]) };
            let mut blocks = [[0i8; 4]; Li::LEN * Ln::LEN / 8];
            arrange_in_blocks(&weight, &mut blocks);
            interleave(&blocks, &mut nn.weight.0, Ln::LEN);
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase];
            cursor += unsafe { copy_bytes(&mut nn.bias.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase].next;
            cursor += unsafe { copy_bytes(&mut nn.weight.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase].next.next;
            let mut weight = [[0f32; Ln::LEN]; Ln::LEN / 2];
            cursor += unsafe { copy_bytes(&mut weight, &bytes[cursor..]) };
            arrange_in_blocks(&weight, &mut nn.weight.0);
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase].next.next;
            cursor += unsafe { copy_bytes(&mut nn.bias.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase].next.next.next;
            let mut weight = [[0f32; Ln::LEN]; Ln::LEN / 2];
            cursor += unsafe { copy_bytes(&mut weight, &bytes[cursor..]) };
            arrange_in_blocks(&weight, &mut nn.weight.0);
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase].next.next.next;
            cursor += unsafe { copy_bytes(&mut nn.bias.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase].next.next.next.next;
            cursor += unsafe { copy_bytes(&mut nn.weight.0, &bytes[cursor..]) };
        }

        for phase in Phase::iter() {
            let nn = &mut nnue.nn[phase].next.next.next.next;
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
    pub fn nn(phase: Phase) -> &'static Lin<Lro<Lnn<Lnn<Lno>>>> {
        &NNUE.nn[phase]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn feature_transformer_does_not_overflow() {
        let transformer = Nnue::transformer();
        (0..Accumulator::LEN).for_each(|i| {
            let bias = transformer.bias[i] as i32;
            let (mut lower, mut upper) = (bias, bias);

            let mut ka = Vec::from_iter(transformer.ka.iter().map(|a| a[i]));
            let mut ti = Vec::from_iter(transformer.ti.iter().map(|a| a[i] as i16));
            let mut pp = Vec::from_iter(transformer.pp.iter().map(|a| a[i] as i16));

            for (n, ws) in [(32, &mut ka), (128, &mut ti), (240, &mut pp)] {
                let (small, _, _) = ws.select_nth_unstable(n);
                small.iter().for_each(|&v| lower += v as i32);

                let len = ws.len();
                let (_, _, large) = ws.select_nth_unstable(len - 1 - n);
                large.iter().for_each(|&v| upper += v as i32);
            }

            assert!((i16::MIN as i32..=i16::MAX as i32).contains(&lower));
            assert!((i16::MIN as i32..=i16::MAX as i32).contains(&upper));
        });
    }
}

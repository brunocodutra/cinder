use crate::chess::Phase;
use crate::util::{Assume, Integer};
use bytemuck::{Zeroable, zeroed};
use byteorder::{LittleEndian, ReadBytesExt};
use ruzstd::decoding::StreamingDecoder;
use std::cell::SyncUnsafeCell;
use std::io::{self, Read};

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

/// An Efficiently Updatable Neural Network.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
pub struct Nnue {
    transformer: Affine<i16, { Accumulator::LEN }>,
    output: [Output<{ Accumulator::LEN }>; Phase::LEN],
}

static NNUE: SyncUnsafeCell<Nnue> = SyncUnsafeCell::new(zeroed());

#[cold]
#[ctor::ctor]
#[inline(never)]
unsafe fn init() {
    let encoded = include_bytes!("nnue/nn.zst").as_slice();
    let decoder = StreamingDecoder::new(encoded).assume();
    unsafe { Nnue::load(NNUE.get().as_mut_unchecked(), decoder).assume() }
}

impl Nnue {
    #[inline(always)]
    fn load<T: Read>(&mut self, mut reader: T) -> io::Result<()> {
        reader.read_i16_into::<LittleEndian>(&mut *self.transformer.bias)?;
        for row in self.transformer.weight.iter_mut() {
            reader.read_i16_into::<LittleEndian>(row)?;
        }

        for Output { bias, .. } in self.output.iter_mut() {
            *bias = reader.read_i32::<LittleEndian>()?;
        }

        for Output { weight, .. } in self.output.iter_mut() {
            for row in weight {
                reader.read_i16_into::<LittleEndian>(row)?;
            }
        }

        Ok(())
    }

    #[inline(always)]
    pub fn transformer() -> &'static Affine<i16, { Accumulator::LEN }> {
        unsafe { &NNUE.get().as_ref_unchecked().transformer }
    }

    #[inline(always)]
    pub fn output(phase: Phase) -> &'static Output<{ Accumulator::LEN }> {
        let idx = phase.cast::<usize>();
        unsafe { NNUE.get().as_ref_unchecked().output.get_unchecked(idx) }
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

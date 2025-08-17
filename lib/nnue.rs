use crate::chess::{Color, Phase, Piece, Role, Square};
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
    positional: Affine<i16, { Positional::LEN }>,
    material: Linear<i32, { Material::LEN }>,
    output: [Output<{ Positional::LEN }>; Material::LEN],
    pieces: [[i32; Role::MAX as usize + 1]; Material::LEN],
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
        reader.read_i16_into::<LittleEndian>(&mut *self.positional.bias)?;
        for row in self.positional.weight.iter_mut() {
            reader.read_i16_into::<LittleEndian>(row)?;
        }

        for row in self.material.weight.iter_mut() {
            reader.read_i32_into::<LittleEndian>(row)?;
        }

        for Output { bias, .. } in self.output.iter_mut() {
            *bias = reader.read_i32::<LittleEndian>()?;
        }

        for Output { weight, .. } in self.output.iter_mut() {
            for row in weight {
                reader.read_i16_into::<LittleEndian>(row)?;
            }
        }

        for phase in Phase::iter() {
            for role in Role::iter() {
                let mut deltas = [0i32, 0i32];

                for sq in Square::iter() {
                    for (d, side) in deltas.iter_mut().zip(Color::iter()) {
                        for ksq in Square::iter() {
                            let feat = Feature::new(side, ksq, Piece::new(role, Color::White), sq);
                            *d += self.material.weight[feat.cast::<usize>()][phase.cast::<usize>()];
                        }
                    }
                }

                self.pieces[phase.cast::<usize>()][role.cast::<usize>()] =
                    (deltas[0] - deltas[1]) / (Square::MAX as i32 + 1).pow(2);
            }
        }

        Ok(())
    }

    #[inline(always)]
    pub fn positional() -> &'static Affine<i16, { Positional::LEN }> {
        unsafe { &NNUE.get().as_ref_unchecked().positional }
    }

    #[inline(always)]
    pub fn material() -> &'static Linear<i32, { Material::LEN }> {
        unsafe { &NNUE.get().as_ref_unchecked().material }
    }

    #[inline(always)]
    pub fn output(phase: Phase) -> &'static Output<{ Positional::LEN }> {
        let idx = phase.cast::<usize>();
        unsafe { NNUE.get().as_ref_unchecked().output.get_unchecked(idx) }
    }

    #[inline(always)]
    pub fn pieces(phase: Phase) -> &'static [i32; Role::MAX as usize + 1] {
        let idx = phase.cast::<usize>();
        unsafe { NNUE.get().as_ref_unchecked().pieces.get_unchecked(idx) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayVec;

    #[test]
    fn feature_transformer_does_not_overflow() {
        (0..Positional::LEN).for_each(|i| {
            let bias = Nnue::positional().bias[i] as i32;
            let mut features = ArrayVec::<_, { Feature::LEN }>::from_iter(
                Nnue::positional().weight.iter().map(|a| a[i] as i32),
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

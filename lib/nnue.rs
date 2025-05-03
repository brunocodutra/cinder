use crate::chess::{Color, Piece, Role, Square};
use crate::util::{Assume, Integer};
use byteorder::{LittleEndian, ReadBytesExt};
use ruzstd::decoding::StreamingDecoder;
use std::cell::SyncUnsafeCell;
use std::io::{self, Read};
use std::mem::{MaybeUninit, transmute};

mod accumulator;
mod evaluator;
mod feature;
mod hidden;
mod transformer;
mod value;

pub use accumulator::*;
pub use evaluator::*;
pub use feature::*;
pub use hidden::*;
pub use transformer::*;
pub use value::*;

/// An [Efficiently Updatable Neural Network][NNUE].
///
/// [NNUE]: https://www.chessprogramming.org/NNUE
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Nnue {
    positional: Affine<i16, { Accumulator::POSITIONAL }>,
    material: Linear<i32, { Accumulator::MATERIAL }>,
    hidden: [Hidden<{ Accumulator::POSITIONAL }>; Accumulator::MATERIAL],
    pieces: [[i32; Role::MAX as usize + 1]; Accumulator::MATERIAL],
}

static NNUE: SyncUnsafeCell<Nnue> = unsafe { MaybeUninit::zeroed().assume_init() };

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
        reader.read_i16_into::<LittleEndian>(unsafe {
            transmute::<
                &mut [[_; Accumulator::POSITIONAL]; Feature::LEN],
                &mut [_; Feature::LEN * Accumulator::POSITIONAL],
            >(&mut *self.positional.weight)
        })?;

        reader.read_i32_into::<LittleEndian>(unsafe {
            transmute::<
                &mut [[_; Accumulator::MATERIAL]; Feature::LEN],
                &mut [_; Feature::LEN * Accumulator::MATERIAL],
            >(&mut *self.material.weight)
        })?;

        for Hidden { bias, weight } in &mut self.hidden {
            *bias = reader.read_i32::<LittleEndian>()?;
            reader.read_i8_into(unsafe {
                transmute::<
                    &mut [[_; Accumulator::POSITIONAL]; 2],
                    &mut [_; Accumulator::POSITIONAL * 2],
                >(weight)
            })?;
        }

        debug_assert!(reader.read_u8().is_err());

        for phase in 0..Accumulator::MATERIAL {
            for role in Role::iter() {
                let mut deltas = [0i32, 0i32];

                for sq in Square::iter() {
                    for (delta, side) in deltas.iter_mut().zip(Color::iter()) {
                        for ksq in Square::iter() {
                            let feat = Feature::new(side, ksq, Piece::new(role, Color::White), sq);
                            *delta += self.material.weight[feat.cast::<usize>()][phase];
                        }
                    }
                }

                self.pieces[phase][role.cast::<usize>()] =
                    (deltas[0] - deltas[1]) / (Square::MAX as i32 + 1).pow(2);
            }
        }

        Ok(())
    }

    #[inline(always)]
    fn positional() -> &'static Affine<i16, { Accumulator::POSITIONAL }> {
        unsafe { &NNUE.get().as_ref_unchecked().positional }
    }

    #[inline(always)]
    fn material() -> &'static Linear<i32, { Accumulator::MATERIAL }> {
        unsafe { &NNUE.get().as_ref_unchecked().material }
    }

    #[inline(always)]
    fn hidden(phase: usize) -> &'static Hidden<{ Accumulator::POSITIONAL }> {
        unsafe { NNUE.get().as_ref_unchecked().hidden.get_unchecked(phase) }
    }

    #[inline(always)]
    fn pieces(phase: usize) -> &'static [i32; Role::MAX as usize + 1] {
        unsafe { NNUE.get().as_ref_unchecked().pieces.get_unchecked(phase) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayVec;

    #[test]
    fn feature_transformer_does_not_overflow() {
        (0..Accumulator::POSITIONAL).for_each(|i| {
            let bias = Nnue::positional().bias[i] as i32;
            let mut features = ArrayVec::<_, { Feature::LEN }>::from_iter(
                Nnue::positional().weight.iter().map(|a| a[i] as i32),
            );

            for weights in features.array_chunks_mut::<768>() {
                let (small, _, _) = weights.select_nth_unstable(32);
                assert!(small.iter().fold(bias, |s, &v| s + v).abs() <= i16::MAX as i32);
                let (_, _, large) = weights.select_nth_unstable(735);
                assert!(large.iter().fold(bias, |s, &v| s + v).abs() <= i16::MAX as i32);
            }
        });
    }
}

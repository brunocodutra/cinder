use byteorder::{LittleEndian, ReadBytesExt};
use ruzstd::decoding::StreamingDecoder;
use std::cell::SyncUnsafeCell;
use std::io::{self, Read};
use std::mem::{MaybeUninit, transmute};

mod accumulator;
mod evaluator;
mod feature;
mod hidden;
mod material;
mod positional;
mod transformer;
mod value;

pub use accumulator::*;
pub use evaluator::*;
pub use feature::*;
pub use hidden::*;
pub use material::*;
pub use positional::*;
pub use transformer::*;
pub use value::*;

/// An [Efficiently Updatable Neural Network][NNUE].
///
/// [NNUE]: https://www.chessprogramming.org/NNUE
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct Nnue {
    ft: Affine<i16, { Positional::LEN }>,
    psqt: Linear<i32, { Material::LEN }>,
    hidden: [Hidden<{ Positional::LEN }>; Material::LEN],
}

static NNUE: SyncUnsafeCell<Nnue> = unsafe { MaybeUninit::zeroed().assume_init() };

#[cold]
#[ctor::ctor]
#[inline(never)]
unsafe fn init() {
    let encoded = include_bytes!("nnue/nn.zst").as_slice();
    let decoder = StreamingDecoder::new(encoded).expect("failed to initialize zstd decoder");
    unsafe { Nnue::load(NNUE.get().as_mut_unchecked(), decoder).expect("failed to load the NNUE") }
}

impl Nnue {
    #[inline(always)]
    fn load<T: Read>(&mut self, mut reader: T) -> io::Result<()> {
        reader.read_i16_into::<LittleEndian>(&mut *self.ft.bias)?;
        reader.read_i16_into::<LittleEndian>(unsafe {
            transmute::<
                &mut [[_; Positional::LEN]; Feature::LEN],
                &mut [_; Feature::LEN * Positional::LEN],
            >(&mut *self.ft.weight)
        })?;

        reader.read_i32_into::<LittleEndian>(unsafe {
            transmute::<
                &mut [[_; Material::LEN]; Feature::LEN],
                &mut [_; Feature::LEN * Material::LEN],
            >(&mut *self.psqt.weight)
        })?;

        for Hidden { bias, weight } in &mut self.hidden {
            *bias = reader.read_i32::<LittleEndian>()?;
            reader.read_i8_into(unsafe {
                transmute::<&mut [[_; Positional::LEN]; 2], &mut [_; Positional::LEN * 2]>(weight)
            })?;
        }

        debug_assert!(reader.read_u8().is_err());

        Ok(())
    }

    #[inline(always)]
    fn psqt() -> &'static Linear<i32, { Material::LEN }> {
        unsafe { &NNUE.get().as_ref_unchecked().psqt }
    }

    #[inline(always)]
    fn ft() -> &'static Affine<i16, { Positional::LEN }> {
        unsafe { &NNUE.get().as_ref_unchecked().ft }
    }

    #[inline(always)]
    fn hidden(phase: usize) -> &'static Hidden<{ Positional::LEN }> {
        unsafe { NNUE.get().as_ref_unchecked().hidden.get_unchecked(phase) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayVec;

    #[test]
    fn feature_transformer_does_not_overflow() {
        (0..Positional::LEN).for_each(|i| {
            let bias = Nnue::ft().bias[i] as i32;
            let mut features = ArrayVec::<_, { Feature::LEN }>::from_iter(
                Nnue::ft().weight.iter().map(|a| a[i] as i32),
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

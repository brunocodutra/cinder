use crate::{nnue::Accumulator, simd::Aligned};

/// Trait for types that represent layers in a neural network.
pub trait Layer {
    /// The number of neurons in this layer.
    const LEN: usize;

    /// The integer type representing each neuron.
    type Neuron;
}

impl<T: Layer> Layer for &T {
    const LEN: usize = T::LEN;
    type Neuron = T::Neuron;
}

impl<I, const N: usize> Layer for [I; N] {
    const LEN: usize = N;
    type Neuron = I;
}

impl<T: Layer> Layer for Aligned<T> {
    const LEN: usize = T::LEN;
    type Neuron = T::Neuron;
}

/// The first hidden layer.
pub type L1<'a> = (&'a Accumulator, &'a Accumulator);

impl<'a> Layer for L1<'a> {
    const LEN: usize = Accumulator::LEN;
    type Neuron = <Accumulator as Layer>::Neuron;
}

/// The n-th hidden layer.
pub type Ln<'a> = &'a Aligned<[f32; 32]>;

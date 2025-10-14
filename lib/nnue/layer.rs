use crate::nnue::Accumulator;
use crate::util::Aligned;

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

/// The perspective neuron layer.
pub type Layer0<'a> = &'a Accumulator;

/// The first neuron layer.
pub type Layer1<'a> = (Layer0<'a>, Layer0<'a>);

/// The second neuron layer.
pub type Layer2<'a> = &'a Aligned<[f32; 16]>;

/// The third neuron layer.
pub type Layer3<'a> = &'a Aligned<[f32; 32]>;

impl<'a> Layer for Layer1<'a> {
    const LEN: usize = Layer0::<'a>::LEN;
    type Neuron = <Layer0<'a> as Layer>::Neuron;
}

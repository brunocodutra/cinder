use crate::nnue::Layer;

/// Trait for types that can connect layers in a neural network.
pub trait Synapse {
    /// The network input.
    type Input<'a>: Layer;

    /// The network output.
    type Output;

    /// Transforms input neurons.
    fn forward<'a>(&self, input: Self::Input<'a>) -> Self::Output;
}

use crate::params::{Param, Params};
use derive_more::with_trait::{Display, Error};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use proptest::prelude::*;

/// The reason why constructing [`Param`] from a floating point failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display("value out of range")]
pub struct ValueOutOfRange;

#[derive(Debug, Default, Copy, Clone, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "spsa", serde(into = "f64"))]
#[cfg_attr(feature = "spsa", serde(try_from = "f64"))]
#[repr(transparent)]
pub struct Constant<const V: i32> {}

impl<const V: i32> Constant<V> {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<const V: i32> Param for Constant<V> {
    const LEN: usize = 0;
    type Value = i32;

    fn get(&self) -> Self::Value {
        V
    }

    fn min() -> Self::Value {
        V
    }

    fn max() -> Self::Value {
        V
    }

    fn perturb<I: IntoIterator<Item = f64>>(&self, _: I) -> (Self, Self) {
        (*self, *self)
    }

    fn update<I: IntoIterator<Item = f64>>(&mut self, _: I) {}
}

impl<const V: i32> From<Constant<V>> for f64 {
    fn from(_: Constant<V>) -> Self {
        V as _
    }
}

impl<const V: i32> TryFrom<f64> for Constant<V> {
    type Error = ValueOutOfRange;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if !(Self::min()..=Self::max()).contains(&(value as i32)) {
            return Err(ValueOutOfRange);
        }

        Ok(Constant {})
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "spsa", serde(into = "f64"))]
#[cfg_attr(feature = "spsa", serde(try_from = "f64"))]
#[repr(transparent)]
pub struct Scalar<const V: i32> {
    #[cfg_attr(test, strategy((Self::min()..=Self::max()).prop_map(|i| i as f64)))]
    value: f64,
}

impl<const V: i32> Scalar<V> {
    pub const fn new() -> Self {
        Self { value: V as f64 }
    }

    pub fn range() -> i32 {
        V.abs().max(Params::BASE / 4)
    }
}

impl<const V: i32> Param for Scalar<V> {
    const LEN: usize = 1;
    type Value = i32;

    fn get(&self) -> Self::Value {
        self.value as _
    }

    fn min() -> Self::Value {
        V - Self::range() / 2
    }

    fn max() -> Self::Value {
        Self::min() + Self::range()
    }

    fn perturb<I: IntoIterator<Item = f64>>(&self, perturbations: I) -> (Self, Self) {
        let mut perturbations = perturbations.into_iter();
        let (mut left, mut right) = (Self::default(), Self::default());
        let (min, max) = (Self::min() as f64, Self::max() as f64);
        let delta = (max - min) * perturbations.next().unwrap();
        left.value = (self.value + delta).clamp(min, max);
        right.value = (self.value - delta).clamp(min, max);

        (left, right)
    }

    fn update<I: IntoIterator<Item = f64>>(&mut self, corrections: I) {
        let mut corrections = corrections.into_iter();
        let (min, max) = (Self::min() as f64, Self::max() as f64);
        let delta = (max - min) * corrections.next().unwrap();
        self.value = (self.value + delta).clamp(min, max);
    }
}

impl<const V: i32> Default for Scalar<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const V: i32> From<Scalar<V>> for f64 {
    fn from(scalar: Scalar<V>) -> Self {
        scalar.value
    }
}

impl<const V: i32> TryFrom<f64> for Scalar<V> {
    type Error = ValueOutOfRange;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let scalar = Scalar { value };
        if !(Self::min()..=Self::max()).contains(&scalar.get()) {
            return Err(ValueOutOfRange);
        }

        Ok(scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    type MockConstant = Constant<400>;
    type MockScalar = Scalar<400>;

    #[proptest]
    fn constant_value_is_always_within_bounds(p: MockConstant) {
        assert!((MockConstant::min()..=MockConstant::max()).contains(&p.get()));
    }

    #[proptest]
    fn constant_has_an_equivalent_value(p: MockConstant) {
        assert_eq!(Constant::try_from(<f64>::from(p)), Ok(p));
    }

    #[proptest]
    fn converting_constant_from_fails_if_value_too_large(
        #[strategy(MockConstant::max() as f64 + 1.0..)] v: f64,
    ) {
        assert_eq!(MockConstant::try_from(v), Err(ValueOutOfRange));
    }

    #[proptest]
    fn converting_constant_from_fails_if_value_too_small(
        #[strategy(..=MockConstant::min() as f64 - 1.0)] v: f64,
    ) {
        assert_eq!(MockConstant::try_from(v), Err(ValueOutOfRange));
    }

    #[proptest]
    fn scalar_value_is_always_within_bounds(p: MockScalar) {
        assert!((MockScalar::min()..=MockScalar::max()).contains(&p.get()));
    }

    #[proptest]
    fn scalar_has_an_equivalent_value(p: MockScalar) {
        assert_eq!(Scalar::try_from(<f64>::from(p)), Ok(p));
    }

    #[proptest]
    fn converting_scalar_succeeds_value_in_range(
        #[strategy(MockScalar::min() as f64..=MockScalar::max() as f64)] v: f64,
    ) {
        assert_ne!(MockScalar::try_from(v), Err(ValueOutOfRange));
    }

    #[proptest]
    fn converting_scalar_from_fails_if_value_too_large(
        #[strategy(MockScalar::max() as f64 + 1.0..)] v: f64,
    ) {
        assert_eq!(MockScalar::try_from(v), Err(ValueOutOfRange));
    }

    #[proptest]
    fn converting_scalar_from_fails_if_value_too_small(
        #[strategy(..=MockScalar::min() as f64 - 1.0)] v: f64,
    ) {
        assert_eq!(MockScalar::try_from(v), Err(ValueOutOfRange));
    }
}

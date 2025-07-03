use crate::params::{Param, Params};
use derive_more::with_trait::{Display, Error};
use serde::{Deserialize, Serialize};
use std::array;

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

macro_rules! define_vector {
    ($vector:ident, $n:expr, $raw:literal) => {
        #[derive(Debug, Copy, Clone, PartialEq)]
        #[cfg_attr(test, derive(test_strategy::Arbitrary))]
        #[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
        #[cfg_attr(feature = "spsa", serde(into = $raw))]
        #[cfg_attr(feature = "spsa", serde(try_from = $raw))]
        #[repr(transparent)]
        pub struct $vector<const V: [i32; $n]> {
            #[cfg_attr(test, strategy((array::from_fn(|i| Self::min()[i]..=Self::max()[i]))
                                        .prop_map(|is| is.map(|i| i as f64))))]
            values: [f64; $n],
        }

        impl<const V: [i32; $n]> $vector<V> {
            pub const fn new() -> Self {
                let mut values = [0.; $n];

                let mut i = values.len();
                while i > 0 {
                    i -= 1;
                    values[i] = V[i] as _;
                }

                Self { values }
            }

            pub fn range() -> [i32; $n] {
                V.map(|v| v.abs().max(Params::BASE / 4))
            }
        }

        impl<const V: [i32; $n]> Param for $vector<V> {
            const LEN: usize = $n;
            type Value = [i32; $n];

            fn get(&self) -> Self::Value {
                self.values.map(|v| v as _)
            }

            fn min() -> Self::Value {
                let range = Self::range();
                array::from_fn(|i| V[i] - range[i] / 2)
            }

            fn max() -> Self::Value {
                let min = Self::min();
                let range = Self::range();
                array::from_fn(|i| min[i] + range[i])
            }

            fn perturb<I: IntoIterator<Item = f64>>(&self, perturbations: I) -> (Self, Self) {
                let mut perturbations = perturbations.into_iter();
                let (mut left, mut right) = (Self::default(), Self::default());
                let (min, max) = (Self::min(), Self::max());

                for i in 0..Self::LEN {
                    let delta = (max[i] as f64 - min[i] as f64) * perturbations.next().unwrap();
                    left.values[i] = (self.values[i] + delta).clamp(min[i] as f64, max[i] as f64);
                    right.values[i] = (self.values[i] - delta).clamp(min[i] as f64, max[i] as f64);
                }

                (left, right)
            }

            fn update<I: IntoIterator<Item = f64>>(&mut self, corrections: I) {
                let mut corrections = corrections.into_iter();
                let (min, max) = (Self::min(), Self::max());

                for i in 0..Self::LEN {
                    let delta = (max[i] as f64 - min[i] as f64) * corrections.next().unwrap();
                    self.values[i] = (self.values[i] + delta).clamp(min[i] as f64, max[i] as f64);
                }
            }
        }

        impl<const V: [i32; $n]> Default for $vector<V> {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<const V: [i32; $n]> From<$vector<V>> for [f64; $n] {
            fn from(vector: $vector<V>) -> Self {
                vector.values
            }
        }

        impl<const V: [i32; $n]> TryFrom<[f64; $n]> for $vector<V> {
            type Error = ValueOutOfRange;

            fn try_from(values: [f64; $n]) -> Result<Self, Self::Error> {
                let vector = $vector { values };
                if !(Self::min()..=Self::max()).contains(&vector.get()) {
                    return Err(ValueOutOfRange);
                }

                Ok(vector)
            }
        }
    };
}

define_vector!(Vector1, 1, "[f64; 1]");
define_vector!(Vector2, 2, "[f64; 2]");
define_vector!(Vector3, 3, "[f64; 3]");
define_vector!(Vector4, 4, "[f64; 4]");
define_vector!(Vector5, 5, "[f64; 5]");
define_vector!(Vector6, 6, "[f64; 6]");
define_vector!(Vector7, 7, "[f64; 7]");
define_vector!(Vector8, 8, "[f64; 8]");
define_vector!(Vector9, 9, "[f64; 9]");
define_vector!(Vector10, 10, "[f64; 10]");
define_vector!(Vector11, 11, "[f64; 11]");
define_vector!(Vector12, 12, "[f64; 12]");
define_vector!(Vector13, 13, "[f64; 13]");
define_vector!(Vector14, 14, "[f64; 14]");
define_vector!(Vector15, 15, "[f64; 15]");
#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    type MockConstant = Constant<400>;
    type MockScalar = Scalar<400>;
    type MockVector = Vector3<{ [0, -400, 400] }>;

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

    #[proptest]
    fn vector_value_is_always_within_bounds(p: MockVector) {
        assert!((MockVector::min()..=MockVector::max()).contains(&p.get()));
    }

    #[proptest]
    fn vector_has_an_equivalent_value(p: MockVector) {
        assert_eq!(Vector3::try_from(<[f64; 3]>::from(p)), Ok(p));
    }

    #[proptest]
    fn converting_vector_succeeds_value_in_range(
        #[strategy([
            MockVector::min()[0] as f64..=MockVector::max()[0] as f64,
            MockVector::min()[1] as f64..=MockVector::max()[1] as f64,
            MockVector::min()[2] as f64..=MockVector::max()[2] as f64])]
        v: [f64; 3],
    ) {
        assert_ne!(MockVector::try_from(v), Err(ValueOutOfRange));
    }

    #[proptest]
    fn converting_vector_from_fails_if_value_too_large(
        #[strategy([
            MockVector::max()[0] as f64 + 1.0..,
            MockVector::max()[1] as f64 + 1.0..,
            MockVector::max()[2] as f64 + 1.0..])]
        v: [f64; 3],
    ) {
        assert_eq!(MockVector::try_from(v), Err(ValueOutOfRange));
    }

    #[proptest]
    fn converting_vector_from_fails_if_value_too_small(
        #[strategy([
            ..=MockVector::min()[0] as f64 - 1.0,
            ..=MockVector::min()[1] as f64 - 1.0,
            ..=MockVector::min()[2] as f64 - 1.0])]
        v: [f64; 3],
    ) {
        assert_eq!(MockVector::try_from(v), Err(ValueOutOfRange));
    }
}

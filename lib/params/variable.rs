use crate::params::Param;
use derive_more::with_trait::{Display, Error};
use serde::{Deserialize, Serialize};
use std::array;

#[cfg(test)]
use proptest::prelude::*;

/// The reason why constructing [`Param`] from a floating point failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display("value out of range")]
pub struct ValueOutOfRange;

macro_rules! define_constant {
    ($name:ident, $n:expr, $raw:literal) => {
        #[derive(Debug, Default, Copy, Clone, PartialEq)]
        #[cfg_attr(test, derive(test_strategy::Arbitrary))]
        #[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
        #[cfg_attr(feature = "spsa", serde(into = $raw))]
        #[cfg_attr(feature = "spsa", serde(try_from = $raw))]
        #[repr(transparent)]
        pub struct $name<const V: [i64; $n]> {}

        impl<const V: [i64; $n]> $name<V> {
            pub const fn new() -> Self {
                Self {}
            }
        }

        impl<const V: [i64; $n]> Param for $name<V> {
            const LEN: usize = 0;
            type Value = [i64; $n];

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

        impl<const V: [i64; $n]> From<$name<V>> for [i64; $n] {
            fn from(_: $name<V>) -> Self {
                V
            }
        }

        impl<const V: [i64; $n]> TryFrom<[i64; $n]> for $name<V> {
            type Error = ValueOutOfRange;

            fn try_from(value: [i64; $n]) -> Result<Self, Self::Error> {
                if !(Self::min()..=Self::max()).contains(&value) {
                    return Err(ValueOutOfRange);
                }

                Ok($name {})
            }
        }
    };
}

define_constant!(Constant1, 1, "[i64; 1]");
define_constant!(Constant2, 2, "[i64; 2]");
define_constant!(Constant3, 3, "[i64; 3]");
define_constant!(Constant4, 4, "[i64; 4]");
define_constant!(Constant5, 5, "[i64; 5]");
define_constant!(Constant6, 6, "[i64; 6]");
define_constant!(Constant7, 7, "[i64; 7]");
define_constant!(Constant8, 8, "[i64; 8]");
define_constant!(Constant9, 9, "[i64; 9]");

macro_rules! define_variable {
    ($name:ident, $n:expr, $raw:literal) => {
        #[derive(Debug, Copy, Clone, PartialEq)]
        #[cfg_attr(test, derive(test_strategy::Arbitrary))]
        #[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
        #[cfg_attr(feature = "spsa", serde(into = $raw))]
        #[cfg_attr(feature = "spsa", serde(try_from = $raw))]
        #[repr(transparent)]
        pub struct $name<const V: [i64; $n]> {
            #[cfg_attr(test, strategy(
                (array::from_fn(|i| Self::min()[i]..=Self::max()[i])).prop_map(|is| is.map(|i| i as f64))))]
            values: [f64; $n],
        }

        impl<const V: [i64; $n]> $name<V> {
            pub const fn new() -> Self {
                let mut values = [0.; $n];

                let mut i = values.len();
                while i > 0 {
                    i -= 1;
                    values[i] = V[i] as _;
                }

                Self { values }
            }

            pub fn range() -> [i64; $n] {
                V.map(i64::abs)
            }
        }

        impl<const V: [i64; $n]> Param for $name<V> {
            const LEN: usize = $n;
            type Value = [i64; $n];

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

        impl<const V: [i64; $n]> Default for $name<V> {
            fn default() -> Self {
                Self::new()
            }
        }

        impl<const V: [i64; $n]> From<$name<V>> for [f64; $n] {
            fn from(variable: $name<V>) -> Self {
                variable.values
            }
        }

        impl<const V: [i64; $n]> TryFrom<[f64; $n]> for $name<V> {
            type Error = ValueOutOfRange;

            fn try_from(values: [f64; $n]) -> Result<Self, Self::Error> {
                let variable = $name { values };
                if !(Self::min()..=Self::max()).contains(&variable.get()) {
                    return Err(ValueOutOfRange);
                }

                Ok(variable)
            }
        }
    };
}

define_variable!(Variable1, 1, "[f64; 1]");
define_variable!(Variable2, 2, "[f64; 2]");
define_variable!(Variable3, 3, "[f64; 3]");
define_variable!(Variable4, 4, "[f64; 4]");
define_variable!(Variable5, 5, "[f64; 5]");
define_variable!(Variable6, 6, "[f64; 6]");
define_variable!(Variable7, 7, "[f64; 7]");
define_variable!(Variable8, 8, "[f64; 8]");
define_variable!(Variable9, 9, "[f64; 9]");

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    type MockConstant = Constant3<{ [0, -400, 600] }>;
    type MockVariable = Variable3<{ [0, -400, 600] }>;

    #[proptest]
    fn constant_value_is_always_within_bounds(p: MockConstant) {
        assert!((MockConstant::min()..=MockConstant::max()).contains(&p.get()));
    }

    #[proptest]
    fn constant_has_an_equivalent_if_value(p: MockConstant) {
        assert_eq!(MockConstant::try_from(<[i64; 3]>::from(p)), Ok(p));
    }

    #[proptest]
    fn converting_constant_from_fails_if_value_too_large(
        #[strategy([
            MockConstant::max()[0] + 1..,
            MockConstant::max()[0] + 1..,
            MockConstant::max()[0] + 1..])]
        v: [i64; 3],
    ) {
        assert_eq!(MockConstant::try_from(v), Err(ValueOutOfRange));
    }

    #[proptest]
    fn converting_constant_from_fails_if_value_too_small(
        #[strategy([
            ..=MockConstant::min()[0] - 1,
            ..=MockConstant::min()[0] - 1,
            ..=MockConstant::min()[0] - 1])]
        v: [i64; 3],
    ) {
        assert_eq!(MockConstant::try_from(v), Err(ValueOutOfRange));
    }

    #[proptest]
    fn variable_has_an_equivalent_value(p: MockVariable) {
        assert_eq!(Variable3::try_from(<[f64; 3]>::from(p)), Ok(p));
    }

    #[proptest]
    fn converting_variable_succeeds_if_value_in_range(
        #[strategy([
            MockVariable::min()[0]..=MockVariable::max()[0],
            MockVariable::min()[1]..=MockVariable::max()[1],
            MockVariable::min()[2]..=MockVariable::max()[2]])]
        is: [i64; 3],
    ) {
        assert_ne!(
            MockVariable::try_from(is.map(|i| i as f64)),
            Err(ValueOutOfRange)
        );
    }

    #[proptest]
    fn converting_variable_from_fails_if_value_too_large(
        #[strategy([
            MockVariable::max()[0] + 1..,
            MockVariable::max()[1] + 1..,
            MockVariable::max()[2] + 1..])]
        is: [i64; 3],
    ) {
        assert_eq!(
            MockVariable::try_from(is.map(|i| i as f64)),
            Err(ValueOutOfRange)
        );
    }

    #[proptest]
    fn converting_variable_from_fails_if_value_too_small(
        #[strategy([
            ..=MockVariable::min()[0] - 1,
            ..=MockVariable::min()[1] - 1,
            ..=MockVariable::min()[2] - 1])]
        is: [i64; 3],
    ) {
        assert_eq!(
            MockVariable::try_from(is.map(|i| i as f64)),
            Err(ValueOutOfRange)
        );
    }
}

use crate::params::Params;
use derive_more::with_trait::{Display, Error};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use proptest::prelude::*;

#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(feature = "spsa", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "spsa", serde(into = "f64"))]
#[cfg_attr(feature = "spsa", serde(try_from = "f64"))]
#[repr(transparent)]
pub struct Param<const V: i32, const K: i32 = 1> {
    #[cfg_attr(test, strategy((Self::min()..=Self::max()).prop_map(|i| i as f64)))]
    value: f64,
}

impl<const V: i32, const K: i32> Param<V, K> {
    pub const fn new() -> Self {
        Self { value: V as f64 }
    }

    pub fn min() -> i32 {
        Ord::max(V - Ord::max(V, Params::BASE) * K / 2, 0)
    }

    pub fn max() -> i32 {
        Self::min() + Ord::max(V, Params::BASE) * K
    }

    pub fn get(&self) -> i32 {
        self.value as i32
    }

    pub fn perturb<I: IntoIterator<Item = f64>>(&self, perturbations: I) -> (Self, Self) {
        let mut perturbations = perturbations.into_iter();
        let (mut left, mut right) = (Self::default(), Self::default());
        let (min, max) = (Self::min() as f64, Self::max() as f64);
        let delta = (max - min) * perturbations.next().unwrap();
        left.value = (self.value + delta).clamp(min, max);
        right.value = (self.value - delta).clamp(min, max);

        (left, right)
    }

    pub fn update<I: IntoIterator<Item = f64>>(&mut self, corrections: I) {
        let mut corrections = corrections.into_iter();
        let (min, max) = (Self::min() as f64, Self::max() as f64);
        let delta = (max - min) * corrections.next().unwrap();
        self.value = (self.value + delta).clamp(min, max);
    }
}

impl<const V: i32, const K: i32> Default for Param<V, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const V: i32, const K: i32> From<Param<V, K>> for f64 {
    fn from(param: Param<V, K>) -> Self {
        param.value
    }
}

/// The reason why constructing [`Param`] from a floating point failed.
#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
#[display("parameter out of range")]
pub struct ParameterOutOfRange;

impl<const V: i32, const K: i32> TryFrom<f64> for Param<V, K> {
    type Error = ParameterOutOfRange;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        let param = Param { value };
        if !(Self::min()..=Self::max()).contains(&param.get()) {
            return Err(ParameterOutOfRange);
        }

        Ok(param)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_strategy::proptest;

    type MockParam = Param<1000>;

    #[proptest]
    fn param_value_is_always_within_bounds(p: MockParam) {
        assert!((MockParam::min()..=MockParam::max()).contains(&p.get()));
    }

    #[proptest]
    fn param_has_an_equivalent_array_of_values(p: MockParam) {
        assert_eq!(Param::try_from(<f64>::from(p)), Ok(p));
    }

    #[proptest]
    fn converting_param_from_array_succeeds_if_all_values_in_range(
        #[strategy(MockParam::min() as f64..=MockParam::max() as f64)] vs: f64,
    ) {
        assert_ne!(MockParam::try_from(vs), Err(ParameterOutOfRange));
    }

    #[proptest]
    fn converting_param_from_array_fails_if_any_value_out_of_range(
        #[strategy(MockParam::max() as f64 + 1.0..)] vs: f64,
    ) {
        assert_eq!(MockParam::try_from(vs), Err(ParameterOutOfRange));
    }
}

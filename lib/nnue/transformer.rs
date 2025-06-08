use crate::nnue::Feature;
use crate::util::{AlignTo64, Assume, Integer};
use derive_more::with_trait::{Debug, Deref, DerefMut};
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// A linear feature transformer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Deref, DerefMut)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(bound(T, T: From<i8>)))]
#[debug("Linear<{N}>")]
pub struct Linear<T, const N: usize> {
    #[cfg_attr(test, map(|vs: [[i8; N]; Feature::LEN]| AlignTo64(vs.map(|v| v.map(T::from)))))]
    pub(super) weight: AlignTo64<[[T; N]; Feature::LEN]>,
}

impl<T, const N: usize> Linear<T, N>
where
    T: Default + Copy + Add<Output = T> + AddAssign + Sub<Output = T> + SubAssign,
{
    /// A fresh accumulator.
    #[inline(always)]
    pub fn fresh(&self) -> [T; N] {
        [Default::default(); N]
    }

    /// Updates the accumulator by adding a feature.
    #[inline(always)]
    pub fn add(&self, a1: Feature, accumulator: &mut [T; N]) {
        let a1 = self.weight.get(a1.cast::<usize>()).assume();
        for (i, y) in accumulator.iter_mut().enumerate() {
            *y += a1[i];
        }
    }

    /// Updates the accumulator by removing a feature and adding another.
    #[inline(always)]
    pub fn sub_add(&self, s1: Feature, a1: Feature, accumulator: &mut [T; N]) {
        let s1 = self.weight.get(s1.cast::<usize>()).assume();
        let a1 = self.weight.get(a1.cast::<usize>()).assume();
        for (i, y) in accumulator.iter_mut().enumerate() {
            *y += a1[i] - s1[i];
        }
    }

    /// Updates the accumulator by removing two features and adding one other.
    #[inline(always)]
    pub fn sub_sub_add(&self, s1: Feature, s2: Feature, a1: Feature, accumulator: &mut [T; N]) {
        let s1 = self.weight.get(s1.cast::<usize>()).assume();
        let s2 = self.weight.get(s2.cast::<usize>()).assume();
        let a1 = self.weight.get(a1.cast::<usize>()).assume();
        for (i, y) in accumulator.iter_mut().enumerate() {
            *y += a1[i] - s1[i] - s2[i];
        }
    }

    /// Updates the accumulator by removing two features and adding two others.
    #[inline(always)]
    pub fn sub_sub_add_add(
        &self,
        s1: Feature,
        s2: Feature,
        a1: Feature,
        a2: Feature,
        accumulator: &mut [T; N],
    ) {
        let s1 = self.weight.get(s1.cast::<usize>()).assume();
        let s2 = self.weight.get(s2.cast::<usize>()).assume();
        let a1 = self.weight.get(a1.cast::<usize>()).assume();
        let a2 = self.weight.get(a2.cast::<usize>()).assume();
        for (i, y) in accumulator.iter_mut().enumerate() {
            *y += a1[i] - s1[i] + a2[i] - s2[i];
        }
    }
}

/// An affine feature transformer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Deref)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(bound(T, T: From<i8>)))]
#[debug("Affine<{N}>")]
pub struct Affine<T, const N: usize> {
    #[cfg_attr(test, map(|vs: [i8; N]| AlignTo64(vs.map(T::from))))]
    pub(super) bias: AlignTo64<[T; N]>,
    #[deref]
    pub(super) weight: Linear<T, N>,
}

impl<T, const N: usize> Affine<T, N>
where
    T: Default + Copy + Add<Output = T> + AddAssign + Sub<Output = T> + SubAssign,
{
    /// A fresh accumulator.
    #[inline(always)]
    pub fn fresh(&self) -> [T; N] {
        *self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::array::uniform3;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn fresh_accumulator_equals_bias(t: Affine<i16, 2>) {
        assert_eq!(t.fresh(), *t.bias);
    }

    #[proptest]
    fn add_updates_accumulator(
        t: Affine<i16, 3>,
        a1: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.add(a1, &mut new);

        let a1 = t.weight[a1.cast::<usize>()];
        assert_eq!(new, [prev[0] + a1[0], prev[1] + a1[1], prev[2] + a1[2]]);
    }

    #[proptest]
    fn add_sub_updates_accumulator(
        t: Affine<i16, 3>,
        a1: Feature,
        s1: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.sub_add(s1, a1, &mut new);

        let a1 = t.weight[a1.cast::<usize>()];
        let s1 = t.weight[s1.cast::<usize>()];

        assert_eq!(
            new,
            [
                prev[0] + a1[0] - s1[0],
                prev[1] + a1[1] - s1[1],
                prev[2] + a1[2] - s1[2],
            ]
        );
    }

    #[proptest]
    fn add_sub_sub_updates_accumulator(
        t: Affine<i16, 3>,
        a1: Feature,
        s1: Feature,
        s2: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.sub_sub_add(s1, s2, a1, &mut new);

        let a1 = t.weight[a1.cast::<usize>()];
        let s1 = t.weight[s1.cast::<usize>()];
        let s2 = t.weight[s2.cast::<usize>()];

        assert_eq!(
            new,
            [
                prev[0] + a1[0] - s1[0] - s2[0],
                prev[1] + a1[1] - s1[1] - s2[1],
                prev[2] + a1[2] - s1[2] - s2[2],
            ]
        );
    }

    #[proptest]
    fn add_add_sub_sub_updates_accumulator(
        t: Affine<i16, 3>,
        a1: Feature,
        a2: Feature,
        s1: Feature,
        s2: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.sub_sub_add_add(s1, s2, a1, a2, &mut new);

        let a1 = t.weight[a1.cast::<usize>()];
        let a2 = t.weight[a2.cast::<usize>()];
        let s1 = t.weight[s1.cast::<usize>()];
        let s2 = t.weight[s2.cast::<usize>()];

        assert_eq!(
            new,
            [
                prev[0] + a1[0] - s1[0] + a2[0] - s2[0],
                prev[1] + a1[1] - s1[1] + a2[1] - s2[1],
                prev[2] + a1[2] - s1[2] + a2[2] - s2[2],
            ]
        );
    }
}

use crate::nnue::Feature;
use crate::util::{AlignTo64, Assume, Integer};
use derive_more::with_trait::{Deref, DerefMut};
use std::ops::{Add, AddAssign, Sub, SubAssign};

#[cfg(test)]
use proptest::{prelude::*, sample::Index};

#[cfg(test)]
use std::ops::Range;

/// A linear feature transformer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Deref, DerefMut)]
pub struct Linear<T, const N: usize> {
    pub(super) weight: AlignTo64<[[T; N]; Feature::LEN]>,
}

#[cfg(test)]
impl<const N: usize> Arbitrary for Box<Linear<i16, N>> {
    type Parameters = Range<i16>;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(Range { start, end }: Self::Parameters) -> Self::Strategy {
        (any::<Index>())
            .prop_map(move |rng| {
                let mut transformer = unsafe { Self::new_zeroed().assume_init() };

                for v in &mut transformer.weight.iter_mut().flatten() {
                    *v = rng.index((end - start) as _) as i16 + start
                }

                transformer
            })
            .no_shrink()
            .boxed()
    }
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
pub struct Affine<T, const N: usize> {
    pub(super) bias: AlignTo64<[T; N]>,
    #[deref]
    pub(super) weight: Linear<T, N>,
}

#[cfg(test)]
impl<const N: usize> Arbitrary for Box<Affine<i16, N>> {
    type Parameters = Range<i16>;
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(range @ Range { start, end }: Self::Parameters) -> Self::Strategy {
        (any_with::<Box<Linear<i16, N>>>(range), any::<Index>())
            .prop_map(move |(linear, rng)| {
                let mut transformer = unsafe { Self::new_zeroed().assume_init() };

                transformer.weight = *linear;
                for v in transformer.bias.iter_mut() {
                    *v = rng.index((end - start) as _) as i16 + start
                }

                transformer
            })
            .no_shrink()
            .boxed()
    }
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
    use test_strategy::proptest;

    #[proptest]
    fn fresh_accumulator_equals_bias(#[any(-128i16..128)] t: Box<Affine<i16, 2>>) {
        assert_eq!(t.fresh(), *t.bias);
    }

    #[proptest]
    fn add_updates_accumulator(
        #[any(-128i16..128)] t: Box<Affine<i16, 3>>,
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
        #[any(-128..128i16)] t: Box<Affine<i16, 3>>,
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
                prev[2] + a1[2] - s1[1],
            ]
        );
    }

    #[proptest]
    fn add_sub_sub_updates_accumulator(
        #[any(-128..128i16)] t: Box<Affine<i16, 3>>,
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
                prev[2] + a1[2] - s1[1] - s2[1],
            ]
        );
    }

    #[proptest]
    fn add_add_sub_sub_updates_accumulator(
        #[any(-128..128i16)] t: Box<Affine<i16, 3>>,
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
                prev[2] + a1[2] - s1[1] + a2[2] - s2[1],
            ]
        );
    }
}

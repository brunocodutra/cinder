use crate::nnue::Feature;
use crate::util::{Aligned, Assume, Integer};
use bytemuck::Zeroable;
use derive_more::with_trait::Debug;
use std::hint::unreachable_unchecked;
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// The NNUE feature transformer.
#[derive(Debug, Clone, Eq, PartialEq, Hash, Zeroable)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(bound(T, T: From<i8>)))]
#[debug("Transformer<{N}>")]
pub struct Transformer<T, const N: usize> {
    #[cfg_attr(test, map(|vs: [i8; N]| Aligned(vs.map(T::from))))]
    pub bias: Aligned<[T; N]>,
    #[cfg_attr(test, map(|vs: [[i8; N]; Feature::LEN]| Aligned(vs.map(|v| v.map(T::from)))))]
    pub weight: Aligned<[[T; N]; Feature::LEN]>,
}

impl<T, const N: usize> Transformer<T, N>
where
    T: Zeroable + Copy + Add<Output = T> + AddAssign + Sub<Output = T> + SubAssign,
{
    /// Refreshes `accumulator`.
    #[inline(always)]
    pub fn refresh(&self, accumulator: &mut [T; N]) {
        *accumulator = *self.bias;
    }

    /// Updates `acc` by adding and removing features.
    #[inline(always)]
    pub fn accumulate_in_place(
        &self,
        acc: &mut [T; N],
        sub: [Option<Feature>; 2],
        add: [Option<Feature>; 2],
    ) {
        match (sub, add) {
            ([Some(s1), None], [None, None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] -= s1[i];
                }
            }

            ([None, None], [Some(a1), None]) => {
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i];
                }
            }

            ([Some(s1), Some(s2)], [None, None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] -= s1[i] + s2[i];
                }
            }

            ([None, None], [Some(a1), Some(a2)]) => {
                let a1 = self.weight.get(a1.cast::<usize>()).assume();
                let a2 = self.weight.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] + a2[i];
                }
            }

            ([Some(s1), None], [Some(a1), None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i];
                }
            }

            ([Some(s1), None], [Some(a1), Some(a2)]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();
                let a2 = self.weight.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] + a2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] - s2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), Some(a2)]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();
                let a2 = self.weight.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    acc[i] += a1[i] - s1[i] + a2[i] - s2[i];
                }
            }

            _ => unsafe { unreachable_unchecked() },
        }
    }

    /// Updates `dst` by adding and removing features from `src`.
    #[inline(always)]
    pub fn accumulate(
        &self,
        src: &[T; N],
        dst: &mut [T; N],
        sub: [Option<Feature>; 2],
        add: [Option<Feature>; 2],
    ) {
        match (sub, add) {
            ([None, None], [None, None]) => {
                *dst = *src;
            }

            ([None, None], [Some(a1), None]) => {
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i];
                }
            }

            ([Some(s1), None], [Some(a1), None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i] - s1[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), None]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i] - s1[i] - s2[i];
                }
            }

            ([Some(s1), Some(s2)], [Some(a1), Some(a2)]) => {
                let s1 = self.weight.get(s1.cast::<usize>()).assume();
                let s2 = self.weight.get(s2.cast::<usize>()).assume();
                let a1 = self.weight.get(a1.cast::<usize>()).assume();
                let a2 = self.weight.get(a2.cast::<usize>()).assume();

                for i in 0..N {
                    dst[i] = src[i] + a1[i] - s1[i] + a2[i] - s2[i];
                }
            }

            _ => unsafe { unreachable_unchecked() },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::array::uniform3;
    use std::fmt::Debug;
    use test_strategy::proptest;

    #[proptest]
    fn fresh_accumulator_equals_bias(
        t: Transformer<i16, 3>,
        #[strategy(uniform3(-128..128i16))] acc: [i16; 3],
    ) {
        let mut acc = Aligned(acc);
        t.refresh(&mut acc);
        assert_eq!(acc, t.bias);
    }

    #[proptest]
    fn updates_accumulator_in_place_by_removing_one_feature(
        t: Transformer<i16, 3>,
        s1: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.accumulate_in_place(&mut new, [Some(s1), None], [None, None]);

        let s1 = t.weight[s1.cast::<usize>()];
        assert_eq!(new, [prev[0] - s1[0], prev[1] - s1[1], prev[2] - s1[2]]);
    }

    #[proptest]
    fn updates_accumulator_in_place_by_adding_one_feature(
        t: Transformer<i16, 3>,
        a1: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.accumulate_in_place(&mut new, [None, None], [Some(a1), None]);

        let a1 = t.weight[a1.cast::<usize>()];
        assert_eq!(new, [prev[0] + a1[0], prev[1] + a1[1], prev[2] + a1[2]]);
    }

    #[proptest]
    fn updates_accumulator_in_place_by_removing_two_features(
        t: Transformer<i16, 3>,
        s1: Feature,
        s2: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.accumulate_in_place(&mut new, [Some(s1), Some(s2)], [None, None]);

        let s1 = t.weight[s1.cast::<usize>()];
        let s2 = t.weight[s2.cast::<usize>()];

        assert_eq!(
            new,
            [
                prev[0] - s1[0] - s2[0],
                prev[1] - s1[1] - s2[1],
                prev[2] - s1[2] - s2[2],
            ]
        );
    }

    #[proptest]
    fn updates_accumulator_in_place_by_adding_two_features(
        t: Transformer<i16, 3>,
        a1: Feature,
        a2: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.accumulate_in_place(&mut new, [None, None], [Some(a1), Some(a2)]);

        let a1 = t.weight[a1.cast::<usize>()];
        let a2 = t.weight[a2.cast::<usize>()];
        assert_eq!(
            new,
            [
                prev[0] + a1[0] + a2[0],
                prev[1] + a1[1] + a2[1],
                prev[2] + a1[2] + a2[2],
            ]
        );
    }

    #[proptest]
    fn updates_accumulator_in_place_by_adding_one_and_removing_one_features(
        t: Transformer<i16, 3>,
        a1: Feature,
        s1: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.accumulate_in_place(&mut new, [Some(s1), None], [Some(a1), None]);

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
    fn updates_accumulator_in_place_by_adding_two_and_removing_one_features(
        t: Transformer<i16, 3>,
        a1: Feature,
        a2: Feature,
        s1: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.accumulate_in_place(&mut new, [Some(s1), None], [Some(a1), Some(a2)]);

        let a1 = t.weight[a1.cast::<usize>()];
        let a2 = t.weight[a2.cast::<usize>()];
        let s1 = t.weight[s1.cast::<usize>()];

        assert_eq!(
            new,
            [
                prev[0] + a1[0] - s1[0] + a2[0],
                prev[1] + a1[1] - s1[1] + a2[1],
                prev[2] + a1[2] - s1[2] + a2[2],
            ]
        );
    }

    #[proptest]
    fn updates_accumulator_in_place_by_adding_one_and_removing_two_features(
        t: Transformer<i16, 3>,
        a1: Feature,
        s1: Feature,
        s2: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.accumulate_in_place(&mut new, [Some(s1), Some(s2)], [Some(a1), None]);

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
    fn updates_accumulator_in_place_by_adding_two_and_removing_two_features(
        t: Transformer<i16, 3>,
        a1: Feature,
        a2: Feature,
        s1: Feature,
        s2: Feature,
        #[strategy(uniform3(-128..128i16))] prev: [i16; 3],
    ) {
        let mut new = prev;
        t.accumulate_in_place(&mut new, [Some(s1), Some(s2)], [Some(a1), Some(a2)]);

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

    #[proptest]
    fn updates_accumulator_by_adding_one_feature(
        t: Transformer<i16, 3>,
        a1: Feature,
        #[strategy(uniform3(-128..128i16))] src: [i16; 3],
        #[strategy(uniform3(-128..128i16))] mut dst: [i16; 3],
    ) {
        t.accumulate(&src, &mut dst, [None, None], [Some(a1), None]);

        let a1 = t.weight[a1.cast::<usize>()];
        assert_eq!(dst, [src[0] + a1[0], src[1] + a1[1], src[2] + a1[2]]);
    }

    #[proptest]
    fn updates_accumulator_by_adding_one_and_removing_one_features(
        t: Transformer<i16, 3>,
        a1: Feature,
        s1: Feature,
        #[strategy(uniform3(-128..128i16))] src: [i16; 3],
        #[strategy(uniform3(-128..128i16))] mut dst: [i16; 3],
    ) {
        t.accumulate(&src, &mut dst, [Some(s1), None], [Some(a1), None]);

        let a1 = t.weight[a1.cast::<usize>()];
        let s1 = t.weight[s1.cast::<usize>()];

        assert_eq!(
            dst,
            [
                src[0] + a1[0] - s1[0],
                src[1] + a1[1] - s1[1],
                src[2] + a1[2] - s1[2],
            ]
        );
    }

    #[proptest]
    fn updates_accumulator_by_adding_one_and_removing_two_features(
        t: Transformer<i16, 3>,
        a1: Feature,
        s1: Feature,
        s2: Feature,
        #[strategy(uniform3(-128..128i16))] src: [i16; 3],
        #[strategy(uniform3(-128..128i16))] mut dst: [i16; 3],
    ) {
        t.accumulate(&src, &mut dst, [Some(s1), Some(s2)], [Some(a1), None]);

        let a1 = t.weight[a1.cast::<usize>()];
        let s1 = t.weight[s1.cast::<usize>()];
        let s2 = t.weight[s2.cast::<usize>()];

        assert_eq!(
            dst,
            [
                src[0] + a1[0] - s1[0] - s2[0],
                src[1] + a1[1] - s1[1] - s2[1],
                src[2] + a1[2] - s1[2] - s2[2],
            ]
        );
    }

    #[proptest]
    fn updates_accumulator_by_adding_two_and_removing_two_features(
        t: Transformer<i16, 3>,
        a1: Feature,
        a2: Feature,
        s1: Feature,
        s2: Feature,
        #[strategy(uniform3(-128..128i16))] src: [i16; 3],
        #[strategy(uniform3(-128..128i16))] mut dst: [i16; 3],
    ) {
        t.accumulate(&src, &mut dst, [Some(s1), Some(s2)], [Some(a1), Some(a2)]);

        let a1 = t.weight[a1.cast::<usize>()];
        let a2 = t.weight[a2.cast::<usize>()];
        let s1 = t.weight[s1.cast::<usize>()];
        let s2 = t.weight[s2.cast::<usize>()];

        assert_eq!(
            dst,
            [
                src[0] + a1[0] - s1[0] + a2[0] - s2[0],
                src[1] + a1[1] - s1[1] + a2[1] - s2[1],
                src[2] + a1[2] - s1[2] + a2[2] - s2[2],
            ]
        );
    }
}

use crate::params::Param;
use derive_more::with_trait::Constructor;

pub type Scalar<const V: i32> = Constant<V>;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Constructor)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Constant<const V: i32> {}

impl<const V: i32> Param for Constant<V> {
    type Value = i32;

    #[inline(always)]
    fn get(&self) -> Self::Value {
        V
    }
}

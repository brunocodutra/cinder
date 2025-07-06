use crate::params::Param;
use derive_more::with_trait::Constructor;

pub type Constant1<const V: [i64; 1]> = Variable1<V>;
pub type Constant2<const V: [i64; 2]> = Variable2<V>;
pub type Constant3<const V: [i64; 3]> = Variable3<V>;
pub type Constant4<const V: [i64; 4]> = Variable4<V>;
pub type Constant5<const V: [i64; 5]> = Variable5<V>;
pub type Constant6<const V: [i64; 6]> = Variable6<V>;
pub type Constant7<const V: [i64; 7]> = Variable7<V>;
pub type Constant8<const V: [i64; 8]> = Variable8<V>;
pub type Constant9<const V: [i64; 9]> = Variable9<V>;

macro_rules! define_variable {
    ($name:ident, $n:expr) => {
        #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Constructor)]
        #[cfg_attr(test, derive(test_strategy::Arbitrary))]
        pub struct $name<const V: [i64; $n]> {}

        impl<const V: [i64; $n]> Param for $name<V> {
            type Value = [i64; $n];

            #[inline(always)]
            fn get(&self) -> Self::Value {
                V
            }
        }
    };
}

define_variable!(Variable1, 1);
define_variable!(Variable2, 2);
define_variable!(Variable3, 3);
define_variable!(Variable4, 4);
define_variable!(Variable5, 5);
define_variable!(Variable6, 6);
define_variable!(Variable7, 7);
define_variable!(Variable8, 8);
define_variable!(Variable9, 9);

use crate::util::Int;
use nom::{branch::*, bytes::complete::*, character::complete::*, combinator::*, multi::*};
use nom::{error::*, sequence::*, *};
use std::str::FromStr;
use std::time::Duration;

pub fn int<I: Int>(input: &str) -> IResult<&str, I> {
    recognize((opt(alt([tag("-"), tag("+")])), digit1))
        .map_res(i128::from_str)
        .map(i128::saturate)
        .parse(input)
}

pub fn millis(input: &str) -> IResult<&str, Duration> {
    int.map(Duration::from_millis).parse(input)
}

pub fn word(input: &str) -> IResult<&str, &str> {
    take_till1(char::is_whitespace).parse(input)
}

pub fn find<'s, O, F>(inner: F) -> impl Parser<&'s str, Output = O, Error = Error<&'s str>>
where
    F: Parser<&'s str, Output = O, Error = Error<&'s str>>,
{
    many_till(value((), anychar), inner).map(|(_, r)| r)
}

pub fn t<'s, O, F>(inner: F) -> impl Parser<&'s str, Output = O, Error = Error<&'s str>>
where
    F: Parser<&'s str, Output = O, Error = Error<&'s str>>,
{
    delimited(multispace0, inner, multispace0)
}

pub fn field<'s, O, V>(
    key: &str,
    value: V,
) -> impl Parser<&'s str, Output = O, Error = Error<&'s str>>
where
    V: Parser<&'s str, Output = O, Error = Error<&'s str>>,
{
    preceded(t(tag(key)), value)
}

#[expect(clippy::type_complexity)]
pub fn gather<'s, A, B, C, D, E, F, G, H, I, J>(
    inner: (A, B, C, D, E, F, G, H, I, J),
) -> impl Parser<
    &'s str,
    Output = (
        Option<<A as Parser<&'s str>>::Output>,
        Option<<B as Parser<&'s str>>::Output>,
        Option<<C as Parser<&'s str>>::Output>,
        Option<<D as Parser<&'s str>>::Output>,
        Option<<E as Parser<&'s str>>::Output>,
        Option<<F as Parser<&'s str>>::Output>,
        Option<<G as Parser<&'s str>>::Output>,
        Option<<H as Parser<&'s str>>::Output>,
        Option<<I as Parser<&'s str>>::Output>,
        Option<<J as Parser<&'s str>>::Output>,
    ),
    Error = Error<&'s str>,
>
where
    A: Parser<&'s str, Error = Error<&'s str>>,
    B: Parser<&'s str, Error = Error<&'s str>>,
    C: Parser<&'s str, Error = Error<&'s str>>,
    D: Parser<&'s str, Error = Error<&'s str>>,
    E: Parser<&'s str, Error = Error<&'s str>>,
    F: Parser<&'s str, Error = Error<&'s str>>,
    G: Parser<&'s str, Error = Error<&'s str>>,
    H: Parser<&'s str, Error = Error<&'s str>>,
    I: Parser<&'s str, Error = Error<&'s str>>,
    J: Parser<&'s str, Error = Error<&'s str>>,
{
    let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h, mut i, mut j) = inner;

    move |input: &'s str| {
        let mut output = (None, None, None, None, None, None, None, None, None, None);

        let a = |s| a.parse(s);
        let b = |s| b.parse(s);
        let c = |s| c.parse(s);
        let d = |s| d.parse(s);
        let e = |s| e.parse(s);
        let f = |s| f.parse(s);
        let g = |s| g.parse(s);
        let h = |s| h.parse(s);
        let i = |s| i.parse(s);
        let j = |s| j.parse(s);

        let inner = alt((
            a.map(|o| output.0 = Some(o)),
            b.map(|o| output.1 = Some(o)),
            c.map(|o| output.2 = Some(o)),
            d.map(|o| output.3 = Some(o)),
            e.map(|o| output.4 = Some(o)),
            f.map(|o| output.5 = Some(o)),
            g.map(|o| output.6 = Some(o)),
            h.map(|o| output.7 = Some(o)),
            i.map(|o| output.8 = Some(o)),
            j.map(|o| output.9 = Some(o)),
        ));

        let (rest, ()) = fold_many0(inner, || (), |(), ()| ()).parse(input)?;
        Ok((rest, output))
    }
}

macro_rules! define_gather {
    ($gather:ident, $recurse:ident, $($i:ident),+) => {
        #[expect(non_snake_case)]
        pub fn $gather<'s, $($i),+>(inner: ($($i),+)) -> impl Parser<
            &'s str,
            Output = ($(Option<<$i as Parser<&'s str>>::Output>),+),
            Error = Error<&'s str>,
        >
        where
            $($i: Parser<&'s str, Error = Error<&'s str>>),+
        {
            let ($($i),+) = inner;
            $recurse(($($i),+, fail::<_, (), _>())).map(|($($i),+, _)| ($($i),+))
        }
    };
}

define_gather!(gather9, gather, A, B, C, D, E, F, G, H, I);
define_gather!(gather8, gather9, A, B, C, D, E, F, G, H);
define_gather!(gather7, gather8, A, B, C, D, E, F, G);
define_gather!(gather6, gather7, A, B, C, D, E, F);
define_gather!(gather5, gather6, A, B, C, D, E);
define_gather!(gather4, gather5, A, B, C, D);
define_gather!(gather3, gather4, A, B, C);
define_gather!(gather2, gather3, A, B);

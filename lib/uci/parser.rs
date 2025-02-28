use crate::util::Integer;
use nom::{branch::*, bytes::complete::*, character::complete::*, combinator::*, multi::*};
use nom::{error::*, sequence::*, *};
use std::time::Duration;

pub fn int(input: &str) -> IResult<&str, i64> {
    recognize((opt(alt([tag("-"), tag("+")])), digit1))
        .map_res(|s: &str| s.parse())
        .parse(input)
}

pub fn millis(input: &str) -> IResult<&str, Duration> {
    int.map(|i| Duration::from_millis(i.saturate()))
        .parse(input)
}

pub fn word(input: &str) -> IResult<&str, &str> {
    take_till1(char::is_whitespace).parse(input)
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

type Gather5Output<OA, OB, OC, OD, OE> =
    (Option<OA>, Option<OB>, Option<OC>, Option<OD>, Option<OE>);

pub fn gather5<'s, OA, OB, OC, OD, OE, A, B, C, D, E>(
    (mut a, mut b, mut c, mut d, mut e): (A, B, C, D, E),
) -> impl Parser<&'s str, Output = Gather5Output<OA, OB, OC, OD, OE>, Error = Error<&'s str>>
where
    A: Parser<&'s str, Output = OA, Error = Error<&'s str>>,
    B: Parser<&'s str, Output = OB, Error = Error<&'s str>>,
    C: Parser<&'s str, Output = OC, Error = Error<&'s str>>,
    D: Parser<&'s str, Output = OD, Error = Error<&'s str>>,
    E: Parser<&'s str, Output = OE, Error = Error<&'s str>>,
{
    move |input: &'s str| {
        let mut output = (None, None, None, None, None);

        let a = |s| a.parse(s);
        let b = |s| b.parse(s);
        let c = |s| c.parse(s);
        let d = |s| d.parse(s);
        let e = |s| e.parse(s);

        let inner = alt((
            a.map(|o| output.0 = Some(o)),
            b.map(|o| output.1 = Some(o)),
            c.map(|o| output.2 = Some(o)),
            d.map(|o| output.3 = Some(o)),
            e.map(|o| output.4 = Some(o)),
        ));

        let (rest, _) = many1(inner).parse(input)?;
        Ok((rest, output))
    }
}

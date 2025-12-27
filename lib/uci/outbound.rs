use crate::search::{HashSize, Info, Mate, Pv, ThreadCount};
use crate::{chess::Move, util::Int};
use std::fmt::{self, Display, Formatter};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum Outbound {
    BestMove(Option<Move>),
    ReadyOk,
    Info {
        depth: Option<u8>,
        time: Option<Duration>,
        nodes: Option<u64>,
        nps: Option<f64>,
        pv: Option<Pv>,
    },
    UciOk,
}

impl From<Info> for Outbound {
    fn from(info: Info) -> Self {
        Outbound::Info {
            depth: Some(info.depth().cast()),
            time: Some(info.time()),
            nodes: Some(info.nodes()),
            nps: Some(info.nps()),
            pv: Some(info.pv().clone()),
        }
    }
}

impl Display for Outbound {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Outbound::BestMove(None) => f.write_str("bestmove 0000"),
            Outbound::BestMove(Some(best)) => write!(f, "bestmove {best}"),
            Outbound::ReadyOk => f.write_str("readyok"),
            Outbound::Info {
                depth,
                time,
                nodes,
                nps,
                pv,
            } => {
                f.write_str("info")?;

                if let Some(d) = depth {
                    write!(f, " depth {d}")?;
                }

                if let Some(t) = time {
                    write!(f, " time {}", t.as_millis())?;
                }

                if let Some(n) = nodes {
                    write!(f, " nodes {n}")?;
                }

                if let Some(nps) = nps {
                    write!(f, " nps {}", *nps as u64)?;
                }

                if let Some(pv) = pv {
                    const NORMALIZE_TO_PAWN_VALUE: i32 = 68;
                    let normalized_score = pv.score().cast::<i32>() * 100 / NORMALIZE_TO_PAWN_VALUE;

                    match pv.score().mate() {
                        Mate::None => write!(f, " score cp {normalized_score}")?,
                        Mate::Mating(p) => write!(f, " score mate {}", (p + 1) / 2)?,
                        Mate::Mated(p) => write!(f, " score mate -{}", (p + 1) / 2)?,
                    }

                    if pv.head().is_some() {
                        write!(f, " pv {}", pv.moves())?;
                    }
                }

                Ok(())
            }

            Outbound::UciOk => {
                writeln!(f, "id name Cinder {}", env!("CARGO_PKG_VERSION"))?;
                writeln!(f, "id author Bruno Dutra")?;

                writeln!(
                    f,
                    "option name Hash type spin default {} min {} max {}",
                    HashSize::default(),
                    HashSize::lower(),
                    HashSize::upper()
                )?;

                writeln!(
                    f,
                    "option name Threads type spin default {} min {} max {}",
                    ThreadCount::default(),
                    ThreadCount::lower(),
                    ThreadCount::upper()
                )?;

                writeln!(f, "option name SyzygyPath type string default <empty>")?;

                f.write_str("uciok")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::parsers::*;
    use nom::error::Error as ParseError;
    use nom::{branch::*, bytes::complete::*, combinator::*, multi::*, *};
    use test_strategy::proptest;

    fn uciok(input: &str) -> IResult<&str, &str, ParseError<&str>> {
        recognize((find(t(tag("uciok"))), eof)).parse(input)
    }

    fn readyok(input: &str) -> IResult<&str, &str, ParseError<&str>> {
        recognize((t(tag("readyok")), eof)).parse(input)
    }

    fn info(input: &str) -> IResult<&str, &str, ParseError<&str>> {
        let depth = field("depth", int::<i64>);
        let time = field("time", int::<i64>);
        let nodes = field("nodes", int::<i64>);
        let nps = field("nps", int::<i64>);
        let score = field("score", (t(alt([tag("cp"), tag("mate")])), int::<i64>));
        let pv = field("pv", separated_list1(tag(" "), word));

        let info = (
            tag("info"),
            opt(depth),
            opt(time),
            opt(nodes),
            opt(nps),
            opt(score),
            opt(pv),
        );

        recognize((info, eof)).parse(input)
    }

    fn bestmove(input: &str) -> IResult<&str, &str, ParseError<&str>> {
        recognize((field("bestmove", t(alt((tag("0000"), word)))), eof)).parse(input)
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn displays_valid_uci_commands(o: Outbound) {
        let mut parser = match o {
            Outbound::UciOk => uciok,
            Outbound::ReadyOk => readyok,
            Outbound::Info { .. } => info,
            Outbound::BestMove(..) => bestmove,
        };

        assert!(parser.parse(&o.to_string()).is_ok());
    }
}

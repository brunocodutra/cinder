use crate::chess::Move;
use crate::search::{HashSize, Info, Mate, Pv, ThreadCount};
use crate::util::Num;
use std::fmt::{self, Display, Formatter};
use std::time::Duration;

#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[allow(clippy::large_enum_variant)]
pub enum Outbound {
    BestMove(Option<Move>),
    ReadyOk,
    Info {
        depth: u8,
        nodes: u64,
        time: Duration,
        pv: Option<Pv>,
    },
    UciOk,
}

impl From<Option<Move>> for Outbound {
    fn from(m: Option<Move>) -> Self {
        Outbound::BestMove(m)
    }
}

impl From<Info> for Outbound {
    fn from(info: Info) -> Self {
        Outbound::Info {
            depth: info.depth().cast(),
            nodes: info.nodes(),
            time: info.time(),
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
                nodes,
                time,
                pv,
            } => {
                let ms = time.as_millis();
                let nps = *nodes as u128 * 1000 / ms.max(1);
                write!(f, "info depth {depth} time {ms} nodes {nodes} nps {nps}")?;

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
        let depth = field("depth", int::<u64>);
        let time = field("time", millis);
        let nodes = field("nodes", int::<u64>);
        let nps = field("nps", int::<u64>);
        let score = field("score", (t(alt([tag("cp"), tag("mate")])), int::<i64>));
        let pv = field("pv", separated_list1(tag(" "), word));
        let info = (tag("info"), depth, time, nodes, nps, opt((score, opt(pv))));
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

        parser.parse(&o.to_string()).expect("is ok");
    }
}

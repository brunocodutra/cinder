use crate::chess::{Move, Position, Square};
use crate::nnue::Evaluator;
use crate::search::{Depth, HashSize, ThreadCount};
use crate::util::{Assume, parsers::*};
use derive_more::with_trait::{Display, Error, From};
use nom::error::Error as ParseError;
use nom::{branch::*, bytes::complete::*, combinator::*, sequence::*, *};
use std::str::{self, FromStr};
use std::{collections::HashSet, io::Write, time::Duration};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct UciMove([u8; 5]);

impl UciMove {
    #[inline(always)]
    fn new(m: Move) -> Self {
        let mut buffer = [b'\0'; 5];
        write!(&mut buffer[..], "{m}").assume();
        Self(buffer)
    }
}

impl PartialEq<str> for UciMove {
    #[inline(always)]
    fn eq(&self, other: &str) -> bool {
        let len = if self.0[4] == b'\0' { 4 } else { 5 };
        other == unsafe { str::from_utf8_unchecked(&self.0[..len]) }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum Inbound {
    Position(Box<Evaluator>),
    Go {
        depth: Option<Depth>,
        nodes: Option<u64>,
        time: Option<Duration>,
        wtime: Option<Duration>,
        btime: Option<Duration>,
        winc: Option<Duration>,
        binc: Option<Duration>,
        mtg: Option<u8>,
        mate: Option<u8>,
    },
    Perft(u8),
    SetOptionHash(HashSize),
    SetOptionThreads(ThreadCount),
    SetOptionSyzygyPath(HashSet<String>),
    IsReady,
    UciNewGame,
    Uci,
    Stop,
    Quit,
}

impl Inbound {
    pub fn go_infinite() -> Self {
        Self::Go {
            depth: None,
            nodes: None,
            time: None,
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            mtg: None,
            mate: None,
        }
    }
}

#[derive(Debug, Display, Clone, Eq, PartialEq, Error, From)]
pub enum ParseUciError<'s> {
    #[display("unrecognized sequence `{}`", _0.input)]
    Unrecognized(#[error(not(source))] ParseError<&'s str>),
    #[display("illegal move `{_0}`")]
    IllegalMove(#[error(not(source))] &'s str),
}

#[derive(Debug, Default)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct UciParser;

impl UciParser {
    #[cfg(unix)]
    const PATH_DELIMITER: &str = ":";

    #[cfg(windows)]
    const PATH_DELIMITER: &str = ";";

    #[inline(always)]
    pub fn parse<'s>(&mut self, s: &'s str) -> Result<Inbound, ParseUciError<'s>> {
        let mut cmd = t(alt((
            tag("position"),
            tag("go"),
            tag("setoption"),
            tag("perft"),
            tag("isready"),
            tag("ucinewgame"),
            tag("uci"),
            tag("stop"),
            tag("quit"),
        )));

        match cmd.parse(s).finish()? {
            (args, "position") => {
                let word6 = (word, t(word), t(word), t(word), t(word), word);
                let fen = field("fen", t(recognize(word6))).map_res(Position::from_str);
                let startpos = t(tag("startpos")).map(|_| Position::default());
                let moves = opt(field("moves", rest));

                let mut position = terminated((alt((startpos, fen)), moves), eof);
                let (_, (pos, moves)) = position.parse(args).finish()?;
                let mut pos = Box::new(Evaluator::new(pos));

                if let Some(moves) = moves {
                    for s in moves.split_ascii_whitespace() {
                        let take2 = take::<_, _, ParseError<&str>>(2usize);
                        let (_, whence) = take2.map_res(Square::from_str).parse(s).finish()?;

                        let moves = pos.moves();
                        let mut moves_iter = moves.unpack_if(|ms| ms.whence() == whence);
                        let Some(m) = moves_iter.find(|m| UciMove::new(*m) == *s) else {
                            return Err(ParseUciError::IllegalMove(s));
                        };

                        pos.push(Some(m));
                        pos.reset();
                    }
                }

                Ok(Inbound::Position(pos))
            }

            (args, "go") => {
                let wtime = field("wtime", millis);
                let winc = field("winc", millis);
                let btime = field("btime", millis);
                let binc = field("binc", millis);
                let time = field("movetime", millis);
                let nodes = field("nodes", int);
                let depth = field("depth", int);
                let mate = field("mate", int);
                let mtg = field("movestogo", int);
                let inf = t(tag("infinite"));

                let params = gather((wtime, winc, btime, binc, time, nodes, depth, mate, mtg, inf));
                let mut go = terminated(opt(params), eof).map(Option::unwrap_or_default);
                let (_, (wtime, winc, btime, binc, time, nodes, depth, mate, mtg, inf)) =
                    go.parse(args).finish()?;

                match inf {
                    Some(_) => Ok(Inbound::go_infinite()),
                    None => Ok(Inbound::Go {
                        depth,
                        nodes,
                        time,
                        wtime,
                        btime,
                        winc,
                        binc,
                        mtg,
                        mate,
                    }),
                }
            }

            (args, "setoption") => {
                let option = |n| preceded((t(tag("name")), tag_no_case(n), t(tag("value"))), rest);

                use Inbound::*;
                let options = alt((
                    option("hash").map_res(str::parse).map(SetOptionHash),
                    option("threads").map_res(str::parse).map(SetOptionThreads),
                    option("syzygypath").map(|s| {
                        SetOptionSyzygyPath(
                            s.split(Self::PATH_DELIMITER).map(str::to_owned).collect(),
                        )
                    }),
                ));

                let mut setoption = terminated(options, eof);
                let (_, uci) = setoption.parse(args).finish()?;
                Ok(uci)
            }

            (args, "perft") => {
                let mut perft = terminated(t(int).map(Inbound::Perft), eof);
                let (_, uci) = perft.parse(args).finish()?;
                Ok(uci)
            }

            ("", "isready") => Ok(Inbound::IsReady),
            ("", "ucinewgame") => Ok(Inbound::UciNewGame),
            ("", "uci") => Ok(Inbound::Uci),
            ("", "stop") => Ok(Inbound::Stop),
            ("", "quit") => Ok(Inbound::Quit),

            #[expect(clippy::unreachable)]
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::{collection::hash_set, prelude::*, sample::Selector};
    use rand::seq::SliceRandom;
    use test_strategy::proptest;

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_position_with_startpos_succeeds(mut p: UciParser) {
        assert_eq!(
            p.parse("position startpos"),
            Ok(Inbound::Position(Box::default()))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_position_with_startpos_and_moves_succeeds(
        mut p: UciParser,
        #[strategy(..=4usize)] n: usize,
        selector: Selector,
    ) {
        let mut input = String::new();
        let mut pos = Position::default();

        input.push_str("position startpos moves");

        for _ in 0..n {
            let m = selector.select(pos.moves().unpack());
            input.push(' ');
            input.push_str(&m.to_string());
            pos.play(m);
        }

        assert_eq!(
            p.parse(&input),
            Ok(Inbound::Position(Box::new(Evaluator::new(pos))))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_position_with_fen_succeeds(mut p: UciParser, pos: Position) {
        assert_eq!(
            p.parse(&format!("position fen {pos}")),
            Ok(Inbound::Position(Box::new(Evaluator::new(pos))))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_position_with_fen_and_moves_succeeds(
        mut p: UciParser,
        mut pos: Position,
        #[strategy(..=4usize)] n: usize,
        selector: Selector,
    ) {
        let mut input = String::new();

        input.push_str(&format!("position fen {pos} moves"));

        for _ in 0..n {
            prop_assume!(pos.outcome().is_none());
            let m = selector.select(pos.moves().unpack());
            input.push(' ');
            input.push_str(&m.to_string());
            pos.play(m);
        }

        assert_eq!(
            p.parse(&input),
            Ok(Inbound::Position(Box::new(Evaluator::new(pos))))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_position_with_invalid_fen_fails(
        mut p: UciParser,
        #[filter(#s.parse::<Position>().is_err())] s: String,
    ) {
        assert!(p.parse(&format!("position fen {s}")).is_err());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_position_with_invalid_move_fails(
        mut p: UciParser,
        #[strategy("[^[:ascii:]]+")] s: String,
    ) {
        assert!(p.parse(&format!("position startpos moves {}", s)).is_err());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_position_with_illegal_move_fails(
        mut p: UciParser,
        #[filter(!Position::default().moves().unpack().any(|m| UciMove::new(m) == *#m.to_string()))]
        m: Move,
    ) {
        assert_eq!(
            p.parse(&format!("position startpos moves {}", m)),
            Err(ParseUciError::IllegalMove(&m.to_string()))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_go_time_left_succeeds(
        mut p: UciParser,
        #[strategy(..10u8)] wt: u8,
        #[strategy(..10u8)] wi: u8,
        #[strategy(..10u8)] bt: u8,
        #[strategy(..10u8)] bi: u8,
        idx: usize,
    ) {
        let mut input = [
            "go".to_string(),
            format!("wtime {wt}"),
            format!("btime {bt}"),
            format!("winc {wi}"),
            format!("binc {bi}"),
        ];

        input[1..].shuffle(&mut rand::rng());

        let idx = idx % input.len();
        let input = input[..=idx].join(" ");

        let get = |name| find(field(name, millis)).parse(&input).ok().map(|(_, t)| t);

        assert_eq!(
            p.parse(&input),
            Ok(Inbound::Go {
                depth: None,
                nodes: None,
                time: None,
                wtime: get("wtime"),
                btime: get("btime"),
                winc: get("winc"),
                binc: get("binc"),
                mtg: None,
                mate: None
            })
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_go_depth_succeeds(mut p: UciParser, d: Depth) {
        assert_eq!(
            p.parse(&format!("go depth {d}")),
            Ok(Inbound::Go {
                depth: Some(d),
                nodes: None,
                time: None,
                wtime: None,
                btime: None,
                winc: None,
                binc: None,
                mtg: None,
                mate: None
            })
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_go_nodes_succeeds(mut p: UciParser, n: u64) {
        assert_eq!(
            p.parse(&format!("go nodes {n}")),
            Ok(Inbound::Go {
                depth: None,
                nodes: Some(n),
                time: None,
                wtime: None,
                btime: None,
                winc: None,
                binc: None,
                mtg: None,
                mate: None
            })
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_go_movetime_succeeds(mut p: UciParser, t: u64) {
        assert_eq!(
            p.parse(&format!("go movetime {t}")),
            Ok(Inbound::Go {
                depth: None,
                nodes: None,
                time: Some(Duration::from_millis(t)),
                wtime: None,
                btime: None,
                winc: None,
                binc: None,
                mtg: None,
                mate: None
            })
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_go_with_no_arguments_succeeds(mut p: UciParser) {
        assert_eq!(p.parse("go"), Ok(Inbound::go_infinite()));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_go_infinite_ignores_any_other_limits(mut p: UciParser, d: Depth, n: u64, t: u64) {
        assert_eq!(
            p.parse(&format!("go depth {d} nodes {n} movetime {t} infinite")),
            Ok(Inbound::go_infinite())
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_go_with_moves_to_go_succeeds(mut p: UciParser, mtg: u8) {
        assert_eq!(
            p.parse(&format!("go movestogo {mtg}")),
            Ok(Inbound::Go {
                depth: None,
                nodes: None,
                time: None,
                wtime: None,
                btime: None,
                winc: None,
                binc: None,
                mtg: Some(mtg),
                mate: None
            })
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_go_with_mate_succeeds(mut p: UciParser, n: u8) {
        assert_eq!(
            p.parse(&format!("go mate {n}")),
            Ok(Inbound::Go {
                depth: None,
                nodes: None,
                time: None,
                wtime: None,
                btime: None,
                winc: None,
                binc: None,
                mtg: None,
                mate: Some(n)
            })
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_perft_succeeds(mut p: UciParser, n: u8) {
        assert_eq!(p.parse(&format!("perft {n}")), Ok(Inbound::Perft(n)));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_stop_succeeds(mut p: UciParser) {
        assert_eq!(p.parse(&"stop"), Ok(Inbound::Stop));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_quit_succeeds(mut p: UciParser) {
        assert_eq!(p.parse(&"quit"), Ok(Inbound::Quit));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_uci_succeeds(mut p: UciParser) {
        assert_eq!(p.parse(&"uci"), Ok(Inbound::Uci));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_new_game_succeeds(mut p: UciParser) {
        assert_eq!(p.parse(&"ucinewgame"), Ok(Inbound::UciNewGame));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_isready_succeeds(mut p: UciParser) {
        assert_eq!(p.parse(&"isready"), Ok(Inbound::IsReady));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_option_hash_succeeds(mut p: UciParser, h: HashSize) {
        assert_eq!(
            p.parse(&format!("setoption name Hash value {h}")),
            Ok(Inbound::SetOptionHash(h >> 20 << 20))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_invalid_hash_size_fails(
        mut p: UciParser,
        #[filter(#s.trim().parse::<HashSize>().is_err())] s: String,
    ) {
        let input = format!("setoption name Hash value {s}");
        assert!(p.parse(&input).is_err());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_option_threads_succeeds(mut p: UciParser, t: ThreadCount) {
        assert_eq!(
            p.parse(&format!("setoption name Threads value {t}")),
            Ok(Inbound::SetOptionThreads(t))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_invalid_thread_count_fails(
        mut p: UciParser,
        #[filter(#s.trim().parse::<ThreadCount>().is_err())] s: String,
    ) {
        let input = format!("setoption name Threads value {s}");
        assert!(p.parse(&input).is_err());
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_option_syzygy_path_succeeds(
        mut p: UciParser,
        #[strategy(hash_set("[^[:ascii:]]{1,10}", 1..10))] syzygy: HashSet<String>,
    ) {
        let paths = Vec::from_iter(syzygy.clone()).join(UciParser::PATH_DELIMITER);

        assert_eq!(
            p.parse(&format!("setoption name SyzygyPath value {paths}")),
            Ok(Inbound::SetOptionSyzygyPath(syzygy))
        );
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn parsing_unsupported_command_fails(mut p: UciParser, #[strategy("[^[:ascii:]]*")] s: String) {
        assert!(p.parse(&s).is_err());
    }
}

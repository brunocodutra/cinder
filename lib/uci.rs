use crate::chess::{Color, Move, Position, Square};
use crate::search::{Engine, HashSize, Info, Limits, Mate, Options, ThreadCount};
use crate::util::{Assume, Integer, parsers::*};
use derive_more::with_trait::{Display, Error, From};
use futures::{prelude::*, select_biased as select, stream::FusedStream};
use nom::error::Error as ParseError;
use nom::{branch::*, bytes::complete::*, combinator::*, sequence::*, *};
use std::fmt::{self, Debug, Formatter};
use std::str::{self, FromStr};
use std::{io::Write, path::PathBuf, time::Instant};

#[cfg(test)]
use proptest::{prelude::*, strategy::LazyJust};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct UciMove(Move);

impl PartialEq<str> for UciMove {
    fn eq(&self, other: &str) -> bool {
        let mut buffer = [b'\0'; 5];
        write!(&mut buffer[..], "{}", self.0).assume();
        let len = if buffer[4] == b'\0' { 4 } else { 5 };
        other == unsafe { str::from_utf8_unchecked(&buffer[..len]) }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct UciSearchInfo(Info);

impl Display for UciSearchInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("info")?;

        write!(f, " depth {}", self.0.depth())?;
        write!(f, " time {}", self.0.time().as_millis())?;
        write!(f, " nodes {}", self.0.nodes())?;
        write!(f, " nps {}", self.0.nps() as u64)?;

        match self.0.score().mate() {
            Mate::None => write!(f, " score cp {}", self.0.score())?,
            Mate::Mating(p) => write!(f, " score mate {}", (p + 1) / 2)?,
            Mate::Mated(p) => write!(f, " score mate -{}", (p + 1) / 2)?,
        }

        if self.0.head().is_some() {
            write!(f, " pv {}", self.0.moves())?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct UciBestMove(Option<Move>);

impl Display for UciBestMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => f.write_str("bestmove 0000"),
            Some(best) => write!(f, "bestmove {best}"),
        }
    }
}

#[derive(Debug, Display, Clone, Eq, PartialEq, Error)]
enum UciError<E> {
    #[display("failed to parse the uci command")]
    ParseError,
    Fatal(E),
}

impl<E> From<ParseError<&str>> for UciError<E> {
    fn from(_: ParseError<&str>) -> Self {
        UciError::ParseError
    }
}

/// A basic UCI server.
#[derive(Debug, Default)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(args = I,
    bound(I: 'static + Debug + Default + Clone, O: 'static + Debug + Default + Clone)))]
pub struct Uci<I, O> {
    #[cfg_attr(test, strategy(Just(args.clone())))]
    input: I,
    #[cfg_attr(test, strategy(LazyJust::new(O::default)))]
    output: O,
    #[cfg_attr(test, strategy(LazyJust::new(move || Engine::with_options(&#options))))]
    engine: Engine,
    options: Options,
    position: Position,
}

impl<I, O> Uci<I, O> {
    #[cfg(unix)]
    const PATH_DELIMITER: char = ':';

    #[cfg(windows)]
    const PATH_DELIMITER: char = ';';

    /// Constructs a new uci server instance.
    pub fn new(input: I, output: O) -> Self {
        Self {
            input,
            output,
            engine: Default::default(),
            options: Default::default(),
            position: Default::default(),
        }
    }
}

impl<I: FusedStream<Item = String> + Unpin, O: Sink<String> + Unpin> Uci<I, O> {
    async fn go(&mut self, limits: Limits) -> Result<bool, UciError<O::Error>> {
        let mut search = self.engine.search(&self.position, limits);
        let mut best = UciBestMove(None);

        loop {
            select! {
                info = search.next() => {
                    match info {
                        None => break,
                        Some(i) => {
                            best = UciBestMove(i.head());
                            let info = UciSearchInfo(i).to_string();
                            self.output.send(info).await.map_err(UciError::Fatal)?;
                        }
                    }
                },

                line = self.input.next() => {
                    match line.as_deref().map(str::trim_ascii) {
                        None | Some("") => continue,
                        Some("quit") => return Ok(false),
                        Some("stop") => { search.abort(); },
                        Some(cmd) => eprintln!("Warning: ignored unsupported command `{cmd}` during search"),
                    }
                }
            }
        }

        let bestmove = best.to_string();
        self.output.send(bestmove).await.map_err(UciError::Fatal)?;

        Ok(true)
    }

    async fn execute(&mut self, input: &str) -> Result<bool, UciError<O::Error>> {
        let mut cmd = t(alt((
            tag("position"),
            tag("go"),
            tag("perft"),
            tag("eval"),
            tag("setoption"),
            tag("isready"),
            tag("ucinewgame"),
            tag("uci"),
            tag("stop"),
            tag("quit"),
        )));

        match cmd.parse(input).finish()? {
            (args, "position") => {
                let word6 = (word, t(word), t(word), t(word), t(word), word);
                let fen = field("fen", t(recognize(word6))).map_res(Position::from_str);
                let startpos = t(tag("startpos")).map(|_| Default::default());
                let moves = opt(field("moves", rest));

                let mut position = terminated((alt((startpos, fen)), moves), eof);
                let (_, (mut pos, moves)) = position.parse(args).finish()?;

                if let Some(moves) = moves {
                    for s in moves.split_ascii_whitespace() {
                        let take2 = take::<_, _, ParseError<&str>>(2usize);
                        let (_, whence) = take2.map_res(Square::from_str).parse(s).finish()?;

                        let moves = pos.moves();
                        let mut moves_iter = moves.unpack_if(|ms| ms.whence() == whence);
                        let Some(m) = moves_iter.find(|m| UciMove(*m) == *s) else {
                            return Err(UciError::ParseError);
                        };

                        pos.play(m);
                    }
                }

                self.position = pos;
                Ok(true)
            }

            (args, "go") => {
                let turn = self.position.turn();

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

                let params = (wtime, winc, btime, binc, time, nodes, depth, mate, mtg, inf);
                let limits = gather(params).map(|(wt, wi, bt, bi, t, n, d, _, _, inf)| {
                    let mut limits = Limits::none();

                    if inf.is_none() {
                        if let (Color::White, Some(clock)) = (turn, wt) {
                            limits = limits.with_clock(clock, wi.unwrap_or_default());
                        }

                        if let (Color::Black, Some(clock)) = (turn, bt) {
                            limits = limits.with_clock(clock, bi.unwrap_or_default());
                        }

                        if let Some(movetime) = t {
                            limits = limits.with_time(movetime);
                        }

                        if let Some(nodes) = n {
                            limits = limits.with_nodes(nodes.saturate())
                        }

                        if let Some(depth) = d {
                            limits = limits.with_depth(depth.saturate())
                        }
                    }

                    limits
                });

                let mut go = terminated(opt(limits), eof).map(|l| l.unwrap_or_default());
                let (_, limits) = go.parse(args).finish()?;
                self.go(limits).await
            }

            (args, "perft") => {
                let depth = t(int).map(|i| i.saturate());
                let mut perft = terminated(depth, eof);
                let (_, depth) = perft.parse(args).finish()?;

                let timer = Instant::now();
                let nodes = self.position.perft(depth);
                let millis = timer.elapsed().as_millis();

                let info = format!(
                    "info time {millis} nodes {nodes} nps {}",
                    nodes as u128 * 1000 / millis.max(1)
                );

                self.output.send(info).await.map_err(UciError::Fatal)?;

                Ok(true)
            }

            (args, "setoption") => {
                let option = |n| preceded((t(tag("name")), tag_no_case(n), t(tag("value"))), rest);

                let options = gather3((
                    option("hash").map_res(|s| s.parse()),
                    option("threads").map_res(|s| s.parse()),
                    option("syzygypath")
                        .map(|s| s.split(Self::PATH_DELIMITER).map(PathBuf::from).collect()),
                ));

                let mut setoption = terminated(options, eof);
                let (_, (hash, threads, syzygy)) = setoption.parse(args).finish()?;

                if let Some(h) = hash {
                    self.options.hash = h;
                }

                if let Some(t) = threads {
                    self.options.threads = t;
                }

                if let Some(p) = syzygy {
                    self.options.syzygy = p;
                }

                self.engine = Engine::with_options(&self.options);

                Ok(true)
            }

            ("", "isready") => {
                let readyok = "readyok".to_string();
                self.output.send(readyok).await.map_err(UciError::Fatal)?;

                Ok(true)
            }

            ("", "ucinewgame") => {
                self.engine = Engine::with_options(&self.options);
                self.position = Default::default();

                Ok(true)
            }

            ("", "uci") => {
                let uciok = "uciok".to_string();
                let name = format!("id name Cinder {}", env!("CARGO_PKG_VERSION"));
                let author = "id author Bruno Dutra".to_string();

                let hash = format!(
                    "option name Hash type spin default {} min {} max {}",
                    HashSize::default(),
                    HashSize::lower(),
                    HashSize::upper()
                );

                let threads = format!(
                    "option name Threads type spin default {} min {} max {}",
                    ThreadCount::default(),
                    ThreadCount::lower(),
                    ThreadCount::upper()
                );

                let syzygy = "option name SyzygyPath type string default <empty>".to_string();

                self.output.send(name).await.map_err(UciError::Fatal)?;
                self.output.send(author).await.map_err(UciError::Fatal)?;
                self.output.send(hash).await.map_err(UciError::Fatal)?;
                self.output.send(threads).await.map_err(UciError::Fatal)?;
                self.output.send(syzygy).await.map_err(UciError::Fatal)?;
                self.output.send(uciok).await.map_err(UciError::Fatal)?;

                Ok(true)
            }

            ("", "stop") => Ok(true),

            ("", "quit") => Ok(false),

            _ => unreachable!(),
        }
    }

    /// Runs the UCI server.
    pub async fn run(&mut self) -> Result<(), O::Error> {
        while let Some(line) = self.input.next().await {
            match line.trim_ascii() {
                "" => continue,
                cmd => match self.execute(cmd).await {
                    Ok(false) => break,
                    Ok(true) => continue,
                    Err(UciError::Fatal(e)) => return Err(e),
                    Err(UciError::ParseError) => {
                        eprintln!("Warning: ignored unrecognized command `{cmd}`")
                    }
                },
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{chess::Position, search::Depth};
    use futures::executor::block_on;
    use nom::{character::complete::line_ending, multi::separated_list1};
    use proptest::sample::Selector;
    use rand::seq::SliceRandom;
    use std::task::{Context, Poll};
    use std::{collections::VecDeque, pin::Pin};
    use test_strategy::proptest;

    #[derive(Debug, Default, Clone, Eq, PartialEq)]
    struct StaticStream(VecDeque<String>);

    impl StaticStream {
        fn new(items: impl IntoIterator<Item = impl ToString>) -> Self {
            Self(items.into_iter().map(|s| s.to_string()).collect())
        }
    }

    impl Stream for StaticStream {
        type Item = String;

        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    impl FusedStream for StaticStream {
        fn is_terminated(&self) -> bool {
            self.0.is_empty()
        }
    }

    type MockUci = Uci<StaticStream, Vec<String>>;

    fn info(input: &str) -> IResult<&str, &str, ParseError<&str>> {
        let depth = field("depth", int);
        let time = field("time", int);
        let nodes = field("nodes", int);
        let nps = field("nps", int);
        let score = field("score", (t(alt([tag("cp"), tag("mate")])), int));
        let pv = field("pv", separated_list1(tag(" "), word));
        let info = (tag("info"), depth, time, nodes, nps, score, opt(pv));
        recognize(separated_list1(line_ending, info)).parse(input)
    }

    #[proptest]
    fn handles_position_with_startpos(
        #[any(StaticStream::new(["position startpos"]))] mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, Default::default());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_position_with_startpos_and_moves(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        mut uci: MockUci,
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

        uci.input = StaticStream::new([input]);
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, pos);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_position_with_fen(
        #[any(StaticStream::new([format!("position fen {}", #pos)]))] mut uci: MockUci,
        pos: Position,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position.to_string(), pos.to_string());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_position_with_fen_and_moves(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        mut uci: MockUci,
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

        uci.input = StaticStream::new([input]);
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position.to_string(), pos.to_string());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_position_with_invalid_fen(
        #[any(StaticStream::new([format!("position fen {}", #_s)]))] mut uci: MockUci,
        #[filter(#_s.parse::<Position>().is_err())] _s: String,
    ) {
        let pos = uci.position.clone();
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, pos);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_position_with_invalid_move(
        #[strategy("[^[:ascii:]]+")] _s: String,
        #[any(StaticStream::new([format!("position startpos moves {}", #_s)]))] mut uci: MockUci,
    ) {
        let pos = uci.position.clone();
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, pos);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_position_with_illegal_move(
        #[filter(!Position::default().moves().unpack().any(|m| UciMove(m) == *#_m.to_string()))]
        _m: Move,
        #[any(StaticStream::new([format!("position startpos moves {}", #_m)]))] mut uci: MockUci,
    ) {
        let pos = uci.position.clone();
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, pos);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_go_time_left(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        mut uci: MockUci,
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
        uci.input = StaticStream::new([input[..=(idx % input.len())].join(" ")]);
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((info, line_ending, bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_depth(
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new([format!("go depth {}", #_d)]))]
        mut uci: MockUci,
        _d: Depth,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((info, line_ending, bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_nodes(
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new([format!("go nodes {}", #_n)]))]
        mut uci: MockUci,
        #[strategy(..1000u64)] _n: u64,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((info, line_ending, bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_time(
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new([format!("go movetime {}", #_ms)]))]
        mut uci: MockUci,
        #[strategy(..10u8)] _ms: u8,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((info, line_ending, bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_infinite(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new(["go infinite"]))]
        mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((info, line_ending, bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_with_no_move(
        #[by_ref]
        #[filter(#uci.position.moves().is_empty())]
        #[any(StaticStream::new(["go"]))]
        mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", tag("0000"));
        let mut pattern = recognize(terminated((info, line_ending, bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_with_moves_to_go(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new([format!("go movestogo {}", #_mtg)]))]
        mut uci: MockUci,
        _mtg: i8,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((info, line_ending, bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_with_mate(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new([format!("go mate {}", #_mate)]))]
        mut uci: MockUci,
        _mate: i8,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((info, line_ending, bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_stop_during_search(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new(["go", "stop"]))]
        mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((info, line_ending, bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_quit_during_search(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new(["go", "quit"]))]
        mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
    }

    #[proptest]
    fn handles_stop(#[any(StaticStream::new(["stop"]))] mut uci: MockUci) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_quit(#[any(StaticStream::new(["quit"]))] mut uci: MockUci) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_perft(
        #[any(StaticStream::new([format!("perft {}", #_d)]))] mut uci: MockUci,
        #[strategy(..4u8)] _d: u8,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));

        let output = uci.output.join("\n");

        let time = field("time", int);
        let nodes = field("nodes", int);
        let nps = field("nps", int);
        let info = (tag("info"), time, nodes, nps);
        let mut pattern = recognize(terminated(info, eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_uci(#[any(StaticStream::new(["uci"]))] mut uci: MockUci) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.concat().ends_with("uciok"));
    }

    #[proptest]
    fn handles_new_game(#[any(StaticStream::new(["ucinewgame"]))] mut uci: MockUci) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, Default::default());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_ready(#[any(StaticStream::new(["isready"]))] mut uci: MockUci) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.output.concat(), "readyok");
    }

    #[proptest]
    fn handles_option_hash(
        #[any(StaticStream::new([format!("setoption name Hash value {}", #h)]))] mut uci: MockUci,
        h: HashSize,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.options.hash, h >> 20 << 20);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_invalid_hash_size(
        #[any(StaticStream::new([format!("setoption name Hash value {}", #_s)]))] mut uci: MockUci,
        #[filter(#_s.trim().parse::<HashSize>().is_err())] _s: String,
    ) {
        let o = uci.options.clone();
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.options, o);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_option_threads(
        #[any(StaticStream::new([format!("setoption name Threads value {}", #t)]))]
        mut uci: MockUci,
        t: ThreadCount,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.options.threads, t);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_invalid_thread_count(
        #[any(StaticStream::new([format!("setoption name Threads value {}", #_s)]))]
        mut uci: MockUci,
        #[filter(#_s.trim().parse::<ThreadCount>().is_err())] _s: String,
    ) {
        let o = uci.options.clone();
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.options, o);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_option_syzygy_path(
        #[any(StaticStream::new([format!("setoption name SyzygyPath value {}", #s)]))]
        mut uci: MockUci,
        #[filter(!#s.trim_ascii().is_empty())] s: String,
    ) {
        let paths = s
            .trim_ascii()
            .split(MockUci::PATH_DELIMITER)
            .map(PathBuf::from)
            .collect();

        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.options.syzygy, paths);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_unsupported_messages(
        #[any(StaticStream::new([#_s]))] mut uci: MockUci,
        #[strategy("[^[:ascii:]]*")] _s: String,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.output.join("\n"), "");
    }
}

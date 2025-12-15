use crate::chess::{Color, Move, Square};
use crate::nnue::Evaluator;
use crate::search::{HashSize, Info, Limits, Mate, ThreadCount};
use crate::util::{Assume, Int, parsers::*};
use derive_more::with_trait::{Display, Error, From};
use futures::{pin_mut, prelude::*, select_biased as select, stream::FusedStream};
use nom::error::Error as ParseError;
use nom::{branch::*, bytes::complete::*, combinator::*, sequence::*, *};
use std::fmt::{self, Debug, Formatter};
use std::io::{self, Write};
use std::str::{self, FromStr};
use std::{collections::HashSet, time::Instant};

#[cfg(test)]
use proptest::{prelude::*, strategy::LazyJust};

#[cfg(test)]
mod mock {
    use super::*;
    use crate::search::{Engine, Options};
    use derive_more::{Deref, DerefMut};
    use std::path::Path;
    use test_strategy::Arbitrary;

    #[derive(Debug, Deref, DerefMut, Arbitrary)]
    pub struct MockEngine {
        #[deref]
        #[deref_mut]
        #[strategy(LazyJust::new(move || Engine::with_options(&#options).unwrap()))]
        delegate: Engine,
        pub options: Options,
    }

    impl MockEngine {
        pub fn new() -> io::Result<Self> {
            Ok(MockEngine {
                delegate: Engine::new()?,
                options: Default::default(),
            })
        }

        pub fn set_hash(&mut self, hash: HashSize) -> io::Result<()> {
            self.options.hash = hash;
            self.delegate.set_hash(hash)
        }

        pub fn set_threads(&mut self, threads: ThreadCount) -> io::Result<()> {
            self.options.threads = threads;
            self.delegate.set_threads(threads)
        }

        pub fn set_syzygy<I: IntoIterator<Item: AsRef<Path>>>(
            &mut self,
            paths: I,
        ) -> io::Result<()> {
            self.options.syzygy = paths.into_iter().map(|p| p.as_ref().into()).collect();
            self.delegate.set_syzygy(&self.options.syzygy)
        }
    }
}

#[cfg(test)]
use mock::MockEngine as Engine;

#[cfg(not(test))]
use crate::search::Engine;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct UciMove([u8; 5]);

impl UciMove {
    #[inline(always)]
    fn new(m: Move) -> Self {
        let mut buffer = [b'\0'; 5];
        write!(&mut buffer[..], "{}", m).assume();
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

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct UciSearchInfo(Info);

impl Display for UciSearchInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("info")?;

        write!(f, " depth {}", self.0.depth())?;
        write!(f, " time {}", self.0.time().as_millis())?;
        write!(f, " nodes {}", self.0.nodes())?;
        write!(f, " nps {}", self.0.nps() as u64)?;

        const NORMALIZE_TO_PAWN_VALUE: i32 = 68;
        let normalized_score = self.0.score().cast::<i32>() * 100 / NORMALIZE_TO_PAWN_VALUE;

        match self.0.score().mate() {
            Mate::None => write!(f, " score cp {}", normalized_score)?,
            Mate::Mating(p) => write!(f, " score mate {}", (p + 1) / 2)?,
            Mate::Mated(p) => write!(f, " score mate -{}", (p + 1) / 2)?,
        }

        if self.0.head().is_some() {
            write!(f, " pv {}", self.0.moves())?;
        }

        Ok(())
    }
}

#[derive(Debug, Hash)]
#[derive_const(Clone, Eq, PartialEq)]
struct UciBestMove(Option<Move>);

impl Display for UciBestMove {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => f.write_str("bestmove 0000"),
            Some(best) => write!(f, "bestmove {best}"),
        }
    }
}

#[derive(Debug, Display, Error, From)]
enum UciError {
    #[display("failed to parse the uci command")]
    ParseError,
    #[display("the uci server encountered a fatal error")]
    Fatal(io::Error),
}

impl const From<ParseError<&str>> for UciError {
    fn from(_: ParseError<&str>) -> Self {
        UciError::ParseError
    }
}

/// A basic UCI server.
#[derive(Debug)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
#[cfg_attr(test, arbitrary(args = I, bound(
    I: 'static + Debug + Default + Clone,
    O: 'static + Debug + Default + Clone,
)))]
pub struct Uci<I, O> {
    #[cfg_attr(test, strategy(Just(args.clone())))]
    input: I,
    #[cfg_attr(test, strategy(LazyJust::new(O::default)))]
    output: O,
    engine: Engine,
    #[cfg_attr(test, map(Evaluator::new))]
    pos: Evaluator,
}

impl<I, O> Uci<I, O> {
    #[cfg(unix)]
    const PATH_DELIMITER: char = ':';

    #[cfg(windows)]
    const PATH_DELIMITER: char = ';';

    /// Constructs a new uci server instance.
    pub fn new(input: I, output: O) -> io::Result<Self> {
        Ok(Self {
            input,
            output,
            engine: Engine::new()?,
            pos: Default::default(),
        })
    }
}

impl<I: FusedStream<Item = String> + Unpin, O: Sink<String, Error = io::Error> + Unpin> Uci<I, O> {
    async fn go(&mut self, limits: Limits) -> Result<bool, UciError> {
        let search = self.engine.search(&mut self.pos, limits);
        pin_mut!(search);

        loop {
            select! {
                info = search.next() => {
                    match info {
                        Some(i) => self.output.send(UciSearchInfo(i).to_string()).await?,
                        None => break,
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

        let bestmove = UciBestMove(search.bestmove()).to_string();
        self.output.send(bestmove).await?;

        Ok(true)
    }

    async fn execute(&mut self, input: &str) -> Result<bool, UciError> {
        let mut cmd = t(alt((
            tag("position"),
            tag("go"),
            tag("perft"),
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
                let fen = field("fen", t(recognize(word6))).map_res(Evaluator::from_str);
                let startpos = t(tag("startpos")).map(|_| Evaluator::default());
                let moves = opt(field("moves", rest));

                let mut position = terminated((alt((startpos, fen)), moves), eof);
                let (_, (mut pos, moves)) = position.parse(args).finish()?;

                if let Some(moves) = moves {
                    for s in moves.split_ascii_whitespace() {
                        let take2 = take::<_, _, ParseError<&str>>(2usize);
                        let (_, whence) = take2.map_res(Square::from_str).parse(s).finish()?;

                        let moves = pos.moves();
                        let mut moves_iter = moves.unpack_if(|ms| ms.whence() == whence);
                        let Some(m) = moves_iter.find(|m| UciMove::new(*m) == *s) else {
                            return Err(UciError::ParseError);
                        };

                        pos.push(Some(m));
                        pos.reset();
                    }
                }

                self.pos = pos;
                Ok(true)
            }

            (args, "go") => {
                let turn = self.pos.turn();

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
                let nodes = self.pos.perft(depth);
                let millis = timer.elapsed().as_millis();

                let info = format!(
                    "info time {millis} nodes {nodes} nps {}",
                    nodes as u128 * 1000 / millis.max(1)
                );

                self.output.send(info).await?;

                Ok(true)
            }

            (args, "setoption") => {
                let option = |n| preceded((t(tag("name")), tag_no_case(n), t(tag("value"))), rest);

                let options = gather3((
                    option("hash").map_res(|s| s.parse()),
                    option("threads").map_res(|s| s.parse()),
                    option("syzygypath").map(|s| s.split(Self::PATH_DELIMITER).collect()),
                ));

                let mut setoption = terminated(options, eof);
                let (_, (hash, threads, syzygy)) = setoption.parse(args).finish()?;

                if let Some(h) = hash {
                    self.engine.set_hash(h)?;
                }

                if let Some(t) = threads {
                    self.engine.set_threads(t)?;
                }

                if let Some(p) = syzygy {
                    self.engine.set_syzygy::<HashSet<_>>(p)?;
                }

                Ok(true)
            }

            ("", "isready") => {
                let readyok = "readyok".to_string();
                self.output.send(readyok).await?;
                Ok(true)
            }

            ("", "ucinewgame") => {
                self.pos = Default::default();
                self.engine.reset();
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

                self.output.send(name).await?;
                self.output.send(author).await?;
                self.output.send(hash).await?;
                self.output.send(threads).await?;
                self.output.send(syzygy).await?;
                self.output.send(uciok).await?;

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
    use derive_more::Deref;
    use futures::executor::block_on;
    use nom::{character::complete::line_ending, multi::separated_list1};
    use proptest::sample::Selector;
    use rand::seq::SliceRandom;
    use std::task::{Context, Poll};
    use std::{collections::VecDeque, path::PathBuf, pin::Pin};
    use test_strategy::proptest;

    #[derive(Debug, Default, Clone, Eq, PartialEq, Hash)]
    struct MockStream(VecDeque<String>);

    impl MockStream {
        fn new(items: impl IntoIterator<Item = impl ToString>) -> Self {
            Self(items.into_iter().map(|s| s.to_string()).collect())
        }
    }

    impl Stream for MockStream {
        type Item = String;

        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    impl FusedStream for MockStream {
        fn is_terminated(&self) -> bool {
            self.0.is_empty()
        }
    }

    #[derive(Debug, Default, Clone, Eq, PartialEq, Deref)]
    struct MockSink(Vec<String>);

    impl Sink<String> for MockSink {
        type Error = io::Error;

        fn poll_ready(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            self.0.poll_ready_unpin(cx).map_err(|_| unreachable!())
        }

        fn start_send(mut self: Pin<&mut Self>, item: String) -> Result<(), Self::Error> {
            self.0.start_send_unpin(item).map_err(|_| unreachable!())
        }

        fn poll_flush(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            self.0.poll_flush_unpin(cx).map_err(|_| unreachable!())
        }

        fn poll_close(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            self.0.poll_close_unpin(cx).map_err(|_| unreachable!())
        }
    }

    type MockUci = Uci<MockStream, MockSink>;

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
        #[any(MockStream::new(["position startpos"]))] mut uci: MockUci,
    ) {
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.pos, Default::default());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_position_with_startpos_and_moves(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        mut uci: MockUci,
        #[strategy(..=4usize)] n: usize,
        selector: Selector,
    ) {
        let mut input = String::new();
        let mut pos = Evaluator::default();

        input.push_str("position startpos moves");

        for _ in 0..n {
            let m = selector.select(pos.moves().unpack());
            input.push(' ');
            input.push_str(&m.to_string());
            pos.push(Some(m));
        }

        uci.input = MockStream::new([input]);
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.pos, pos);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_position_with_fen(
        #[any(MockStream::new([format!("position fen {}", #pos)]))] mut uci: MockUci,
        pos: Position,
    ) {
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.pos.to_string(), pos.to_string());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_position_with_fen_and_moves(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
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

        uci.input = MockStream::new([input]);
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.pos.to_string(), pos.to_string());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_position_with_invalid_fen(
        #[any(MockStream::new([format!("position fen {}", #_s)]))] mut uci: MockUci,
        #[filter(#_s.parse::<Position>().is_err())] _s: String,
    ) {
        let pos = uci.pos.clone();
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.pos, pos);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_position_with_invalid_move(
        #[any(MockStream::new([format!("position startpos moves {}", #_s)]))] mut uci: MockUci,
        #[strategy("[^[:ascii:]]+")] _s: String,
    ) {
        let pos = uci.pos.clone();
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.pos, pos);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_position_with_illegal_move(
        #[any(MockStream::new([format!("position startpos moves {}", #_m)]))] mut uci: MockUci,
        #[filter(!Position::default().moves().unpack().any(|m| UciMove::new(m) == *#_m.to_string()))]
        _m: Move,
    ) {
        let pos = uci.pos.clone();
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.pos, pos);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_go_time_left(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
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
        uci.input = MockStream::new([input[..=(idx % input.len())].join(" ")]);
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((opt((info, line_ending)), bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_depth(
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([format!("go depth {}", #_d)]))]
        mut uci: MockUci,
        _d: Depth,
    ) {
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((opt((info, line_ending)), bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_nodes(
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([format!("go nodes {}", #_n)]))]
        mut uci: MockUci,
        #[strategy(..1000u64)] _n: u64,
    ) {
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((opt((info, line_ending)), bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_time(
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([format!("go movetime {}", #_ms)]))]
        mut uci: MockUci,
        #[strategy(..10u8)] _ms: u8,
    ) {
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((opt((info, line_ending)), bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_infinite(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new(["go infinite"]))]
        mut uci: MockUci,
    ) {
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((opt((info, line_ending)), bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_with_no_move(
        #[by_ref]
        #[filter(#uci.pos.moves().is_empty())]
        #[any(MockStream::new(["go"]))]
        mut uci: MockUci,
    ) {
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", tag("0000"));
        let mut pattern = recognize(terminated((opt((info, line_ending)), bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_with_moves_to_go(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([format!("go movestogo {}", #_mtg)]))]
        mut uci: MockUci,
        _mtg: i8,
    ) {
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((opt((info, line_ending)), bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_go_with_mate(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([format!("go mate {}", #_mate)]))]
        mut uci: MockUci,
        _mate: i8,
    ) {
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((opt((info, line_ending)), bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_stop_during_search(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new(["go", "stop"]))]
        mut uci: MockUci,
    ) {
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let bestmove = field("bestmove", word);
        let mut pattern = recognize(terminated((opt((info, line_ending)), bestmove), eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_quit_during_search(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new(["go", "quit"]))]
        mut uci: MockUci,
    ) {
        assert!(block_on(uci.run()).is_ok());
    }

    #[proptest]
    fn handles_stop(#[any(MockStream::new(["stop"]))] mut uci: MockUci) {
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_quit(#[any(MockStream::new(["quit"]))] mut uci: MockUci) {
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_perft(
        #[any(MockStream::new([format!("perft {}", #_d)]))] mut uci: MockUci,
        #[strategy(..4u8)] _d: u8,
    ) {
        assert!(block_on(uci.run()).is_ok());

        let output = uci.output.join("\n");

        let time = field("time", int);
        let nodes = field("nodes", int);
        let nps = field("nps", int);
        let info = (tag("info"), time, nodes, nps);
        let mut pattern = recognize(terminated(info, eof));
        assert_eq!(pattern.parse(&*output).finish(), Ok(("", &*output)));
    }

    #[proptest]
    fn handles_uci(#[any(MockStream::new(["uci"]))] mut uci: MockUci) {
        assert!(block_on(uci.run()).is_ok());
        assert!(uci.output.concat().ends_with("uciok"));
    }

    #[proptest]
    fn handles_new_game(#[any(MockStream::new(["ucinewgame"]))] mut uci: MockUci) {
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.pos, Default::default());
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_ready(#[any(MockStream::new(["isready"]))] mut uci: MockUci) {
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.output.concat(), "readyok");
    }

    #[proptest]
    fn handles_option_hash(
        #[any(MockStream::new([format!("setoption name Hash value {}", #h)]))] mut uci: MockUci,
        h: HashSize,
    ) {
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.engine.options.hash, h >> 20 << 20);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_invalid_hash_size(
        #[any(MockStream::new([format!("setoption name Hash value {}", #_s)]))] mut uci: MockUci,
        #[filter(#_s.trim().parse::<HashSize>().is_err())] _s: String,
    ) {
        let o = uci.engine.options.clone();
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.engine.options, o);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_option_threads(
        #[any(MockStream::new([format!("setoption name Threads value {}", #t)]))] mut uci: MockUci,
        t: ThreadCount,
    ) {
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.engine.options.threads, t);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_invalid_thread_count(
        #[any(MockStream::new([format!("setoption name Threads value {}", #_s)]))] mut uci: MockUci,
        #[filter(#_s.trim().parse::<ThreadCount>().is_err())] _s: String,
    ) {
        let o = uci.engine.options.clone();
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.engine.options, o);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn handles_option_syzygy_path(
        #[any(MockStream::new([format!("setoption name SyzygyPath value {}", #s)]))]
        mut uci: MockUci,
        #[filter(!#s.trim_ascii().is_empty())] s: String,
    ) {
        let mut paths = HashSet::new();
        for s in s.trim_ascii().split(MockUci::PATH_DELIMITER) {
            let path = PathBuf::from(s);
            prop_assume!(!path.exists());
            paths.insert(path);
        }

        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.engine.options.syzygy, paths);
        assert_eq!(uci.output.join("\n"), "");
    }

    #[proptest]
    fn ignores_unsupported_messages(
        #[any(MockStream::new([#_s]))] mut uci: MockUci,
        #[strategy("[^[:ascii:]]*")] _s: String,
    ) {
        assert!(block_on(uci.run()).is_ok());
        assert_eq!(uci.output.join("\n"), "");
    }
}

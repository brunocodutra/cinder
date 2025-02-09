use crate::chess::{Color, Move, Perspective};
use crate::nnue::Evaluator;
use crate::search::{Engine, HashSize, Limits, Options, ThreadCount};
use crate::util::{Assume, Integer, Trigger};
use futures::channel::oneshot::channel as oneshot;
use futures::{future::FusedFuture, prelude::*, select_biased as select, stream::FusedStream};
use std::time::{Duration, Instant};
use std::{fmt::Debug, io::Write, mem::transmute, str, thread};

#[cfg(test)]
use proptest::{prelude::*, strategy::LazyJust};

/// Runs the provided closure on a thread where blocking is acceptable.
///
/// # Safety
///
/// Must be awaited on through completion strictly before any
/// of the variables `f` may capture is dropped.
#[must_use]
unsafe fn unblock<F, R>(f: F) -> impl FusedFuture<Output = R>
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let (tx, rx) = oneshot();
    thread::spawn(transmute::<
        Box<dyn FnOnce() + Send>,
        Box<dyn FnOnce() + Send + 'static>,
    >(Box::new(move || tx.send(f()).assume()) as _));
    rx.map(Assume::assume)
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct UciMove(Move);

impl PartialEq<str> for UciMove {
    fn eq(&self, other: &str) -> bool {
        let mut buffer = [b'\0'; 5];
        write!(&mut buffer[..], "{}", self.0).assume();
        let len = if buffer[4] == b'\0' { 4 } else { 5 };
        other == unsafe { str::from_utf8_unchecked(&buffer[..len]) }
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
    position: Evaluator,
}

impl<I, O> Uci<I, O> {
    /// Constructs a new uci server instance.
    pub fn new(input: I, output: O) -> Self {
        Self {
            input,
            output,
            engine: Engine::default(),
            options: Options::default(),
            position: Evaluator::default(),
        }
    }
}

impl<I: FusedStream<Item = String> + Unpin, O: Sink<String> + Unpin> Uci<I, O> {
    async fn go(&mut self, limits: &Limits) -> Result<(), O::Error> {
        let stopper = Trigger::armed();

        let mut search =
            unsafe { unblock(|| self.engine.search(&self.position, limits, &stopper)) };

        let pv = loop {
            select! {
                pv = search => break pv,
                line = self.input.next() => {
                    match line.as_deref().map(str::trim) {
                        None => break search.await,
                        Some("stop") => { stopper.disarm(); },
                        Some(cmd) => eprintln!("ignored unsupported command `{cmd}` during search"),
                    }
                }
            }
        };

        let info = match pv.score().mate() {
            Some(p) if p > 0 => format!("info score mate {} pv {}", (p + 1) / 2, pv.moves()),
            Some(p) => format!("info score mate {} pv {}", (p - 1) / 2, pv.moves()),
            None => format!("info score cp {:+} pv {}", pv.score(), pv.moves()),
        };

        self.output.send(info).await?;

        if let Some(m) = pv.head() {
            self.output.send(format!("bestmove {m}")).await?;
        }

        Ok(())
    }

    async fn bench(&mut self, limits: &Limits) -> Result<(), O::Error> {
        let stopper = Trigger::armed();
        let timer = Instant::now();
        self.engine.search(&self.position, limits, &stopper);
        let millis = timer.elapsed().as_millis();

        let info = match limits {
            Limits::Depth(d) => format!("info time {millis} depth {d}"),
            Limits::Nodes(nodes) => format!(
                "info time {millis} nodes {nodes} nps {}",
                *nodes as u128 * 1000 / millis
            ),
            _ => return Ok(()),
        };

        self.output.send(info).await
    }

    /// Runs the UCI server.
    pub async fn run(&mut self) -> Result<(), O::Error> {
        while let Some(line) = self.input.next().await {
            match line.split_whitespace().collect::<Vec<_>>().as_slice() {
                ["quit"] => return Ok(()),
                [] | ["stop"] => continue,

                ["go", "wtime", wtime, "btime", btime, "winc", winc, "binc", binc]
                | ["go", "wtime", wtime, "winc", winc, "btime", btime, "binc", binc] => {
                    let (t, i) = match self.position.turn() {
                        Color::White => (wtime, winc),
                        Color::Black => (btime, binc),
                    };

                    match (t.parse(), i.parse()) {
                        (Err(e), _) | (_, Err(e)) => eprintln!("{e}"),
                        (Ok(t), Ok(i)) => {
                            let t = Duration::from_millis(t);
                            let i = Duration::from_millis(i);
                            self.go(&Limits::Clock(t, i)).await?;
                        }
                    }
                }

                ["go", "movetime", time] => match time.parse() {
                    Ok(ms) => self.go(&Duration::from_millis(ms).into()).await?,
                    Err(e) => eprintln!("{e}"),
                },

                ["go", "depth", depth] => match depth.parse() {
                    Ok(d) => self.go(&Limits::Depth(d)).await?,
                    Err(e) => eprintln!("{e}"),
                },

                ["go", "nodes", nodes] => match nodes.parse() {
                    Ok(n) => self.go(&Limits::Nodes(n)).await?,
                    Err(e) => eprintln!("{e}"),
                },

                ["go"] | ["go", "infinite"] => self.go(&Limits::None).await?,

                ["bench", "depth", depth] => match depth.parse() {
                    Ok(d) => self.bench(&Limits::Depth(d)).await?,
                    Err(e) => eprintln!("{e}"),
                },

                ["bench", "nodes", nodes] => match nodes.parse() {
                    Ok(n) => self.bench(&Limits::Nodes(n)).await?,
                    Err(e) => eprintln!("{e}"),
                },

                ["position", "fen", fen @ ..] => match fen.join(" ").parse() {
                    Err(e) => eprintln!("{e}"),
                    Ok(pos) => self.position = pos,
                },

                ["position", "startpos"] => self.position = Evaluator::default(),
                ["position", "startpos", "moves", moves @ ..] => {
                    self.position = Evaluator::default();

                    for s in moves.iter() {
                        let whence = match s[..s.ceil_char_boundary(2)].parse() {
                            Ok(whence) => whence,
                            Err(e) => {
                                eprintln!("invalid move `{s}`, {e}");
                                break;
                            }
                        };

                        let moves = self.position.moves().filter(|ms| ms.whence() == whence);
                        let Some(m) = moves.flatten().find(|m| UciMove(*m) == **s) else {
                            eprintln!("illegal move `{s}` in position `{}`", self.position);
                            break;
                        };

                        self.position.play(m);
                    }
                }

                ["eval"] => {
                    let pos = &self.position;
                    let turn = self.position.turn();
                    let value = pos.evaluate().perspective(turn);
                    let info = format!("info value {value:+}");
                    self.output.send(info).await?;
                }

                ["uci"] => {
                    let name = "id name Cinder".to_string();
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

                    self.output.send(name).await?;
                    self.output.send(author).await?;
                    self.output.send(hash).await?;
                    self.output.send(threads).await?;
                    self.output.send("uciok".to_string()).await?;
                }

                ["ucinewgame"] => {
                    self.engine = Engine::with_options(&self.options);
                    self.position = Evaluator::default();
                }

                ["isready"] => self.output.send("readyok".to_string()).await?,

                ["setoption", "name", "hash", "value", hash]
                | ["setoption", "name", "Hash", "value", hash] => match hash.parse() {
                    Err(e) => eprintln!("{e}"),
                    Ok(h) => {
                        self.options.hash = h;
                        self.engine = Engine::with_options(&self.options);
                    }
                },

                ["setoption", "name", "threads", "value", threads]
                | ["setoption", "name", "Threads", "value", threads] => match threads.parse() {
                    Err(e) => eprintln!("{e}"),
                    Ok(t) => {
                        self.options.threads = t;
                        self.engine = Engine::with_options(&self.options);
                    }
                },

                cmd => eprintln!("ignored unsupported command `{}`", cmd.join(" ")),
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
    use proptest::sample::Selector;
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

    #[proptest]
    fn handles_position_with_startpos(
        #[any(StaticStream::new(["position startpos"]))] mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, Evaluator::default());
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn handles_position_with_fen(
        #[any(StaticStream::new([format!("position fen {}", #pos)]))] mut uci: MockUci,
        pos: Evaluator,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position.to_string(), pos.to_string());
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn ignores_invalid_fen(
        #[any(StaticStream::new([format!("position fen {}", #_s)]))] mut uci: MockUci,
        #[filter(#_s.parse::<Evaluator>().is_err())] _s: String,
    ) {
        let pos = uci.position.clone();
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, pos);
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn handles_position_with_moves(#[strategy(..=4usize)] n: usize, selector: Selector) {
        let mut uci = MockUci::default();
        let mut input = String::new();
        let mut pos = Evaluator::default();

        input.push_str("position startpos moves");

        for _ in 0..n {
            let m = selector.select(pos.moves().flatten());
            input.push(' ');
            input.push_str(&m.to_string());
            pos.play(m);
        }

        uci.input = StaticStream::new([input]);
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, pos);
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn handles_position_with_invalid_move(
        #[strategy("[^[:ascii:]]+")] _s: String,
        #[any(StaticStream::new([format!("position startpos moves {}", #_s)]))] mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, Evaluator::default());
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn handles_position_with_illegal_move(
        #[filter(!Position::default().moves().flatten().any(|m| UciMove(m) == *#_m.to_string()))]
        _m: Move,
        #[any(StaticStream::new([format!("position startpos moves {}", #_m)]))] mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, Evaluator::default());
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn handles_go_time_left(
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new([format!("go wtime {} btime {} winc {} binc {}", #_wt, #_bt, #_wi, #_bi)]))]
        mut uci: MockUci,
        #[strategy(..10u8)] _wt: u8,
        #[strategy(..10u8)] _wi: u8,
        #[strategy(..10u8)] _bt: u8,
        #[strategy(..10u8)] _bi: u8,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.concat().contains("bestmove"));
    }

    #[proptest]
    fn handles_go_depth(
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new([format!("go depth {}", #_d)]))]
        mut uci: MockUci,
        _d: Depth,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.concat().contains("bestmove"));
    }

    #[proptest]
    fn handles_go_nodes(
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new([format!("go nodes {}", #_n)]))]
        mut uci: MockUci,
        #[strategy(..1000u64)] _n: u64,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.concat().contains("bestmove"));
    }

    #[proptest]
    fn handles_go_time(
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new([format!("go movetime {}", #_ms)]))]
        mut uci: MockUci,
        #[strategy(..10u8)] _ms: u8,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.concat().contains("bestmove"));
    }

    #[proptest]
    fn handles_go(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new(["go"]))]
        mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.concat().contains("bestmove"));
    }

    #[proptest]
    fn handles_stop_during_search(
        #[by_ref]
        #[filter(#uci.position.outcome().is_none())]
        #[any(StaticStream::new(["go", "stop"]))]
        mut uci: MockUci,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.concat().contains("bestmove"));
    }

    #[proptest]
    fn handles_stop(#[any(StaticStream::new(["stop"]))] mut uci: MockUci) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn handles_quit(#[any(StaticStream::new(["quit"]))] mut uci: MockUci) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn handles_eval(#[any(StaticStream::new(["eval"]))] mut uci: MockUci) {
        let pos = uci.position.clone();
        let value = match pos.turn() {
            Color::White => pos.evaluate(),
            Color::Black => -pos.evaluate(),
        };

        let value = format!("value {:+}", value);
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.concat().ends_with(&value));
    }

    #[proptest]
    fn handles_uci(#[any(StaticStream::new(["uci"]))] mut uci: MockUci) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.concat().ends_with("uciok"));
    }

    #[proptest]
    fn handles_new_game(#[any(StaticStream::new(["ucinewgame"]))] mut uci: MockUci) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.position, Evaluator::default());
        assert!(uci.output.is_empty());
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
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn ignores_invalid_hash_size(
        #[any(StaticStream::new([format!("setoption name Hash value {}", #_s)]))] mut uci: MockUci,
        #[filter(#_s.trim().parse::<HashSize>().is_err())] _s: String,
    ) {
        let o = uci.options.clone();
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.options, o);
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn handles_option_threads(
        #[any(StaticStream::new([format!("setoption name Threads value {}", #t)]))]
        mut uci: MockUci,
        t: ThreadCount,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert_eq!(uci.options.threads, t);
        assert!(uci.output.is_empty());
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
        assert!(uci.output.is_empty());
    }

    #[proptest]
    fn ignores_unsupported_messages(
        #[any(StaticStream::new([#_s]))] mut uci: MockUci,
        #[strategy("[^[:ascii:]]*")] _s: String,
    ) {
        assert_eq!(block_on(uci.run()), Ok(()));
        assert!(uci.output.is_empty());
    }
}

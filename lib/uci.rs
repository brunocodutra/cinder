mod inbound;
mod outbound;

pub use inbound::*;
pub use outbound::*;

use crate::search::{Depth, Engine, Limits};
use crate::{chess::Color, nnue::Evaluator, util::Num, warn};
use futures::{prelude::*, select_biased as select, stream::FusedStream};
use std::time::{Duration, Instant};
use std::{fmt::Debug, pin::Pin, str::FromStr};

#[cfg(test)]
use proptest::{prelude::*, strategy::LazyJust};

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
    #[cfg_attr(test, map(|pos| Box::new(Evaluator::new(pos))))]
    pos: Box<Evaluator>,
}

impl<I, O> Uci<I, O> {
    /// Constructs a new uci server instance.
    pub fn new(input: I, output: O) -> Self {
        Self {
            input,
            output,
            engine: Engine::new(),
            pos: Default::default(),
        }
    }
}

impl<I, O> Uci<I, O>
where
    I: FusedStream<Item = Inbound> + Unpin,
    O: Sink<Outbound> + Unpin,
{
    /// Runs the UCI server.
    pub async fn run(&mut self) -> Result<(), O::Error> {
        'quit: while let Some(inbound) = self.input.next().await {
            match inbound {
                Inbound::Position(pos) => self.pos = pos,

                Inbound::UciNewGame => {
                    *self.pos = Evaluator::default();
                    self.engine.reset();
                }

                Inbound::Go {
                    depth,
                    nodes,
                    time,
                    wtime,
                    btime,
                    winc,
                    binc,
                    ..
                } => {
                    let (t, i) = match self.pos.turn() {
                        Color::White => (wtime, winc),
                        Color::Black => (btime, binc),
                    };

                    let limits = Limits {
                        depth,
                        nodes,
                        time,
                        clock: match (t, i) {
                            (None, None) => None,
                            (t, i) => Some((t.unwrap_or_default(), i.unwrap_or_default())),
                        },
                    };

                    let mut search = self.engine.search(&self.pos, limits);
                    let mut pinned = unsafe { Pin::new_unchecked(&mut search) };

                    loop {
                        select! {
                            info = pinned.next() => match info {
                                Some(i) => self.output.send(i.into()).await?,
                                None => break,
                            },

                            inbound = self.input.next() => match inbound {
                                None => continue,
                                Some(Inbound::Quit) => break 'quit,
                                Some(Inbound::Stop) => pinned.abort(),
                                _ => warn!("ignored unexpected command"),
                            }
                        }
                    }

                    #[expect(clippy::drop_non_drop)]
                    drop(pinned);

                    let info = search.conclude();
                    let bestmove = info.pv().head();
                    self.output.send(info.into()).await?;
                    self.output.send(bestmove.into()).await?;
                }

                Inbound::Perft(plies) => {
                    let timer = Instant::now();
                    let nodes = self.pos.perft(plies);
                    let time = timer.elapsed();

                    let info = Outbound::Info {
                        time,
                        depth: plies,
                        seldepth: plies.cast(),
                        nodes,
                        tbhits: 0,
                        pv: None,
                    };

                    self.output.send(info).await?;
                }

                Inbound::Bench { depth } => {
                    const FENS: &[&str] = &[
                        "r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14",
                        "4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24",
                        "r3qbrk/6p1/2b2pPp/p3pP1Q/PpPpP2P/3P1B2/2PB3K/R5R1 w - - 16 42",
                        "6k1/1R3p2/6p1/2Bp3p/3P2q1/P7/1P2rQ1K/5R2 b - - 4 44",
                        "8/8/1p2k1p1/3p3p/1p1P1P1P/1P2PK2/8/8 w - - 3 54",
                        "7r/2p3k1/1p1p1qp1/1P1Bp3/p1P2r1P/P7/4R3/Q4RK1 w - - 0 36",
                        "r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10",
                        "3r3k/2r4p/1p1b3q/p4P2/P2Pp3/1B2P3/3BQ1RP/6K1 w - - 3 87",
                        "2r4r/1p4k1/1Pnp4/3Qb1pq/8/4BpPp/5P2/2RR1BK1 w - - 0 42",
                        "4q1bk/6b1/7p/p1p4p/PNPpP2P/KN4P1/3Q4/4R3 b - - 0 37",
                        "2q3r1/1r2pk2/pp3pp1/2pP3p/P1Pb1BbP/1P4Q1/R3NPP1/4R1K1 w - - 2 34",
                        "1r2r2k/1b4q1/pp5p/2pPp1p1/P3Pn2/1P1B1Q1P/2R3P1/4BR1K b - - 1 37",
                        "r3kbbr/pp1n1p1P/3ppnp1/q5N1/1P1pP3/P1N1B3/2P1QP2/R3KB1R b KQkq b3 0 17",
                        "8/6pk/2b1Rp2/3r4/1R1B2PP/P5K1/8/2r5 b - - 16 42",
                        "1r4k1/4ppb1/2n1b1qp/pB4p1/1n1BP1P1/7P/2PNQPK1/3RN3 w - - 8 29",
                        "8/p2B4/PkP5/4p1pK/4Pb1p/5P2/8/8 w - - 29 68",
                        "3r4/ppq1ppkp/4bnp1/2pN4/2P1P3/1P4P1/PQ3PBP/R4K2 b - - 2 20",
                        "5rr1/4n2k/4q2P/P1P2n2/3B1p2/4pP2/2N1P3/1RR1K2Q w - - 1 49",
                        "1r5k/2pq2p1/3p3p/p1pP4/4QP2/PP1R3P/6PK/8 w - - 1 51",
                        "q5k1/5ppp/1r3bn1/1B6/P1N2P2/BQ2P1P1/5K1P/8 b - - 2 34",
                        "r1b2k1r/5n2/p4q2/1ppn1Pp1/3pp1p1/NP2P3/P1PPBK2/1RQN2R1 w - - 0 22",
                        "r1bqk2r/pppp1ppp/5n2/4b3/4P3/P1N5/1PP2PPP/R1BQKB1R w KQkq - 0 5",
                        "r1bqr1k1/pp1p1ppp/2p5/8/3N1Q2/P2BB3/1PP2PPP/R3K2n b Q - 1 12",
                        "r1bq2k1/p4r1p/1pp2pp1/3p4/1P1B3Q/P2B1N2/2P3PP/4R1K1 b - - 2 19",
                        "r4qk1/6r1/1p4p1/2ppBbN1/1p5Q/P7/2P3PP/5RK1 w - - 2 25",
                        "r7/6k1/1p6/2pp1p2/7Q/8/p1P2K1P/8 w - - 0 32",
                        "r3k2r/ppp1pp1p/2nqb1pn/3p4/4P3/2PP4/PP1NBPPP/R2QK1NR w KQkq - 1 5",
                        "3r1rk1/1pp1pn1p/p1n1q1p1/3p4/Q3P3/2P5/PP1NBPPP/4RRK1 w - - 0 12",
                        "5rk1/1pp1pn1p/p3Brp1/8/1n6/5N2/PP3PPP/2R2RK1 w - - 2 20",
                        "8/1p2pk1p/p1p1r1p1/3n4/8/5R2/PP3PPP/4R1K1 b - - 3 27",
                        "8/4pk2/1p1r2p1/p1p4p/Pn5P/3R4/1P3PP1/4RK2 w - - 1 33",
                        "8/5k2/1pnrp1p1/p1p4p/P6P/4R1PK/1P3P2/4R3 b - - 1 38",
                        "8/8/1p1kp1p1/p1pr1n1p/P6P/1R4P1/1P3PK1/1R6 b - - 15 45",
                        "8/8/1p1k2p1/p1prp2p/P2n3P/6P1/1P1R1PK1/4R3 b - - 5 49",
                        "8/8/1p4p1/p1p2k1p/P2npP1P/4K1P1/1P6/3R4 w - - 6 54",
                        "8/8/1p4p1/p1p2k1p/P2n1P1P/4K1P1/1P6/6R1 b - - 6 59",
                        "8/5k2/1p4p1/p1pK3p/P2n1P1P/6P1/1P6/4R3 b - - 14 63",
                        "8/1R6/1p1K1kp1/p6p/P1p2P1P/6P1/1Pn5/8 w - - 0 67",
                        "1rb1rn1k/p3q1bp/2p3p1/2p1p3/2P1P2N/PP1RQNP1/1B3P2/4R1K1 b - - 4 23",
                        "4rrk1/pp1n1pp1/q5p1/P1pP4/2n3P1/7P/1P3PB1/R1BQ1RK1 w - - 3 22",
                        "r2qr1k1/pb1nbppp/1pn1p3/2ppP3/3P4/2PB1NN1/PP3PPP/R1BQR1K1 w - - 4 12",
                        "2r2k2/8/4P1R1/1p6/8/P4K1N/7b/2B5 b - - 0 55",
                        "6k1/5pp1/8/2bKP2P/2P5/p4PNb/B7/8 b - - 1 44",
                        "2rqr1k1/1p3p1p/p2p2p1/P1nPb3/2B1P3/5P2/1PQ2NPP/R1R4K w - - 3 25",
                        "r1b2rk1/p1q1ppbp/6p1/2Q5/8/4BP2/PPP3PP/2KR1B1R b - - 2 14",
                        "6r1/5k2/p1b1r2p/1pB1p1p1/1Pp3PP/2P1R1K1/2P2P2/3R4 w - - 1 36",
                        "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
                        "2rr2k1/1p4bp/p1q1p1p1/4Pp1n/2PB4/1PN3P1/P3Q2P/2RR2K1 w - f6 0 20",
                        "3br1k1/p1pn3p/1p3n2/5pNq/2P1p3/1PN3PP/P2Q1PB1/4R1K1 w - - 0 23",
                        "2r2b2/5p2/5k2/p1r1pP2/P2pB3/1P3P2/K1P3R1/7R w - - 23 93",
                    ];

                    #[cfg(not(test))]
                    let depth = depth.unwrap_or_else(|| Depth::new(16));

                    #[cfg(test)]
                    let depth = depth.unwrap_or_else(Depth::upper);

                    let mut time = Duration::ZERO;
                    let mut seldepth = 0;
                    let mut tbhits = 0;
                    let mut nodes = 0;

                    for fen in FENS {
                        self.engine.reset();
                        let pos = Evaluator::from_str(fen).unwrap();

                        let timer = Instant::now();
                        let mut search = self.engine.search(&pos, Limits::depth(depth));
                        let mut pinned = unsafe { Pin::new_unchecked(&mut search) };

                        let mut info = None;
                        while let i @ Some(_) = pinned.next().await {
                            info = i;
                        }

                        time += timer.elapsed();
                        nodes += info.map_or(0, |i| i.nodes());
                        tbhits += info.map_or(0, |i| i.tbhits());
                        seldepth = info.map_or(0, |i| i.seldepth()).max(seldepth);
                    }

                    let info = Outbound::Info {
                        time,
                        depth: depth.cast(),
                        seldepth,
                        nodes,
                        tbhits,
                        pv: None,
                    };

                    self.output.send(info).await?;
                }

                Inbound::SetOptionHash(hash) => self.engine.set_hash(hash),
                Inbound::SetOptionThreads(threads) => self.engine.set_threads(threads),
                Inbound::SetOptionSyzygyPath(paths) => self.engine.set_syzygy(paths),
                Inbound::IsReady => self.output.send(Outbound::ReadyOk).await?,
                Inbound::Uci => self.output.send(Outbound::UciOk).await?,
                Inbound::Quit => break 'quit,
                Inbound::Stop => continue,
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{HashSize, ThreadCount};
    use futures::executor::block_on;
    use std::assert_matches;
    use std::collections::{HashSet, VecDeque};
    use std::task::{Context, Poll};
    use test_strategy::proptest;

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    struct MockStream(VecDeque<Inbound>);

    impl MockStream {
        fn new<I: IntoIterator<Item = Inbound>>(items: I) -> Self {
            Self(VecDeque::from_iter(items))
        }
    }

    impl Stream for MockStream {
        type Item = Inbound;

        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Ready(self.0.pop_front())
        }
    }

    impl FusedStream for MockStream {
        fn is_terminated(&self) -> bool {
            self.0.is_empty()
        }
    }

    type MockSink = Vec<Outbound>;
    type MockUci = Uci<MockStream, MockSink>;

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_position(
        #[any(MockStream::new([Inbound::Position(Default::default())]))] mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(uci.pos, Default::default());
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_time_left(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        mut uci: MockUci,
        #[strategy(..10u64)] wt: u64,
        #[strategy(..10u64)] wi: u64,
        #[strategy(..10u64)] bt: u64,
        #[strategy(..10u64)] bi: u64,
    ) {
        uci.input = MockStream::new([Inbound::Go {
            depth: None,
            nodes: None,
            time: None,
            wtime: Some(Duration::from_millis(wt)),
            btime: Some(Duration::from_millis(bt)),
            winc: Some(Duration::from_millis(wi)),
            binc: Some(Duration::from_millis(bi)),
            mtg: None,
            mate: None,
        }]);

        block_on(uci.run()).expect("is ok");
        assert_matches!(uci.output.last(), Some(Outbound::BestMove(Some(..))));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_time_left_with_only_increment(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        mut uci: MockUci,
        #[strategy(..10u64)] wi: u64,
        #[strategy(..10u64)] bi: u64,
    ) {
        uci.input = MockStream::new([Inbound::Go {
            depth: None,
            nodes: None,
            time: None,
            wtime: None,
            btime: None,
            winc: Some(Duration::from_millis(wi)),
            binc: Some(Duration::from_millis(bi)),
            mtg: None,
            mate: None,
        }]);

        block_on(uci.run()).expect("is ok");
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_time_left_with_no_increment(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        mut uci: MockUci,
        #[strategy(..10u64)] wt: u64,
        #[strategy(..10u64)] bt: u64,
    ) {
        uci.input = MockStream::new([Inbound::Go {
            depth: None,
            nodes: None,
            time: None,
            wtime: Some(Duration::from_millis(wt)),
            btime: Some(Duration::from_millis(bt)),
            winc: None,
            binc: None,
            mtg: None,
            mate: None,
        }]);

        block_on(uci.run()).expect("is ok");
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_time_left_and_moves_to_go(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        mut uci: MockUci,
        #[strategy(..10u64)] wt: u64,
        #[strategy(..10u64)] wi: u64,
        #[strategy(..10u64)] bt: u64,
        #[strategy(..10u64)] bi: u64,
        mtg: u8,
    ) {
        uci.input = MockStream::new([Inbound::Go {
            depth: None,
            nodes: None,
            time: None,
            wtime: Some(Duration::from_millis(wt)),
            btime: Some(Duration::from_millis(bt)),
            winc: Some(Duration::from_millis(wi)),
            binc: Some(Duration::from_millis(bi)),
            mtg: Some(mtg),
            mate: None,
        }]);

        block_on(uci.run()).expect("is ok");
        assert_matches!(uci.output.last(), Some(Outbound::BestMove(Some(..))));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_depth(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        mut uci: MockUci,
        d: Depth,
    ) {
        uci.input = MockStream::new([Inbound::Go {
            depth: Some(d),
            nodes: None,
            time: None,
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            mtg: None,
            mate: None,
        }]);

        block_on(uci.run()).expect("is ok");
        assert_matches!(uci.output.last(), Some(Outbound::BestMove(Some(..))));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_nodes(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        mut uci: MockUci,
        #[strategy(..1000u64)] n: u64,
    ) {
        uci.input = MockStream::new([Inbound::Go {
            depth: None,
            nodes: Some(n),
            time: None,
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            mtg: None,
            mate: None,
        }]);

        block_on(uci.run()).expect("is ok");
        assert_matches!(uci.output.last(), Some(Outbound::BestMove(Some(..))));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_mate(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        mut uci: MockUci,
        mate: u8,
    ) {
        uci.input = MockStream::new([Inbound::Go {
            depth: None,
            nodes: None,
            time: None,
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            mtg: None,
            mate: Some(mate),
        }]);

        block_on(uci.run()).expect("is ok");
        assert_matches!(uci.output.last(), Some(Outbound::BestMove(Some(..))));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_movetime(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        mut uci: MockUci,
        #[strategy(..10u64)] t: u64,
    ) {
        uci.input = MockStream::new([Inbound::Go {
            depth: None,
            nodes: None,
            time: Some(Duration::from_millis(t)),
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            mtg: None,
            mate: None,
        }]);

        block_on(uci.run()).expect("is ok");
        assert_matches!(uci.output.last(), Some(Outbound::BestMove(Some(..))));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_infinite(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([Inbound::go_infinite()]))]
        mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
        assert_matches!(uci.output.last(), Some(Outbound::BestMove(Some(..))));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_go_with_no_move(
        #[by_ref]
        #[filter(#uci.pos.moves().is_empty())]
        #[any(MockStream::new([Inbound::go_infinite()]))]
        mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
        assert_matches!(uci.output.last(), Some(Outbound::BestMove(None)));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_stop(#[any(MockStream::new([Inbound::Stop]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_stop_during_search(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([Inbound::go_infinite(), Inbound::Stop]))]
        mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
        assert_matches!(uci.output.last(), Some(Outbound::BestMove(Some(..))));
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_quit(#[any(MockStream::new([Inbound::Quit]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_quit_during_search(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([Inbound::go_infinite(), Inbound::Quit]))]
        mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_bench(mut uci: MockUci, d: Option<Depth>) {
        uci.input = MockStream::new([Inbound::Bench { depth: d }]);

        block_on(uci.run()).expect("is ok");
        assert_matches!(&*uci.output, [Outbound::Info { .. }]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_perft(mut uci: MockUci, #[strategy(..3u8)] p: u8) {
        uci.input = MockStream::new([Inbound::Perft(p)]);

        block_on(uci.run()).expect("is ok");
        assert_matches!(&*uci.output, [Outbound::Info { .. }]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_uci(#[any(MockStream::new([Inbound::Uci]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[Outbound::UciOk]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_new_game(#[any(MockStream::new([Inbound::UciNewGame]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_ready(#[any(MockStream::new([Inbound::IsReady]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[Outbound::ReadyOk]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_option_hash(mut uci: MockUci, h: HashSize) {
        uci.input = MockStream::new([Inbound::SetOptionHash(h)]);

        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_option_threads(mut uci: MockUci, t: ThreadCount) {
        uci.input = MockStream::new([Inbound::SetOptionThreads(t)]);

        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest(cases = 1)]
    #[cfg_attr(miri, ignore)]
    fn handles_option_syzygy_path(mut uci: MockUci, ps: HashSet<String>) {
        uci.input = MockStream::new([Inbound::SetOptionSyzygyPath(ps)]);

        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }
}

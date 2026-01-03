mod inbound;
mod outbound;

pub use inbound::*;
pub use outbound::*;

use crate::search::{Engine, Limits};
use crate::{chess::Color, nnue::Evaluator, warn};
use futures::{prelude::*, select_biased as select, stream::FusedStream};
use std::{fmt::Debug, pin::Pin, time::Instant};

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
                    let clock = match self.pos.turn() {
                        Color::White => Option::zip(wtime, winc),
                        Color::Black => Option::zip(btime, binc),
                    };

                    let limits = Limits {
                        depth,
                        nodes,
                        time,
                        clock,
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
                        depth: Some(plies),
                        time: Some(time),
                        nodes: Some(nodes),
                        nps: Some(nodes as f64 / time.as_secs_f64().max(1E-6)),
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
    use crate::search::{Depth, HashSize, ThreadCount};
    use futures::executor::block_on;
    use std::collections::{HashSet, VecDeque};
    use std::task::{Context, Poll};
    use std::time::Duration;
    use test_strategy::proptest;

    #[derive(Debug, Default, Clone, Eq, PartialEq)]
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

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_position(
        #[any(MockStream::new([Inbound::Position(Default::default())]))] mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(uci.pos, Default::default());
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest]
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
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest]
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
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest]
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
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest]
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
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest]
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
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest]
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
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_go_infinite(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([Inbound::go_infinite()]))]
        mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_go_with_no_move(
        #[by_ref]
        #[filter(#uci.pos.moves().is_empty())]
        #[any(MockStream::new([Inbound::go_infinite()]))]
        mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
        assert!(matches!(uci.output.last(), Some(Outbound::BestMove(None))));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_stop(#[any(MockStream::new([Inbound::Stop]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_stop_during_search(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([Inbound::go_infinite(), Inbound::Stop]))]
        mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
        assert!(matches!(
            uci.output.last(),
            Some(Outbound::BestMove(Some(..)))
        ));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_quit(#[any(MockStream::new([Inbound::Quit]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_quit_during_search(
        #[by_ref]
        #[filter(#uci.pos.outcome().is_none())]
        #[any(MockStream::new([Inbound::go_infinite(), Inbound::Quit]))]
        mut uci: MockUci,
    ) {
        block_on(uci.run()).expect("is ok");
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_perft(mut uci: MockUci, #[strategy(..3u8)] p: u8) {
        uci.input = MockStream::new([Inbound::Perft(p)]);

        block_on(uci.run()).expect("is ok");
        assert!(matches!(&*uci.output, [Outbound::Info { .. }]));
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_uci(#[any(MockStream::new([Inbound::Uci]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[Outbound::UciOk]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_new_game(#[any(MockStream::new([Inbound::UciNewGame]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_ready(#[any(MockStream::new([Inbound::IsReady]))] mut uci: MockUci) {
        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[Outbound::ReadyOk]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_option_hash(mut uci: MockUci, h: HashSize) {
        uci.input = MockStream::new([Inbound::SetOptionHash(h)]);

        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_option_threads(mut uci: MockUci, t: ThreadCount) {
        uci.input = MockStream::new([Inbound::SetOptionThreads(t)]);

        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }

    #[proptest]
    #[cfg_attr(miri, ignore)]
    fn handles_option_syzygy_path(mut uci: MockUci, ps: HashSet<String>) {
        uci.input = MockStream::new([Inbound::SetOptionSyzygyPath(ps)]);

        block_on(uci.run()).expect("is ok");
        assert_eq!(&*uci.output, &[]);
    }
}

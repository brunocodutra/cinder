use cinder::uci::Uci;
use futures::{channel::mpsc::unbounded, executor::block_on, sink::unfold as sink};
use std::io::{prelude::*, stdin, stdout};
use std::{future::ready, thread};

fn main() {
    let (tx, input) = unbounded();

    thread::spawn(move || {
        let mut lines = stdin().lock().lines();
        while let Some(Ok(line)) = lines.next() {
            if tx.unbounded_send(line).is_err() {
                break;
            }
        }
    });

    let mut stdout = stdout().lock();
    let output = sink((), |_, line: String| ready(writeln!(stdout, "{line}")));
    block_on(Uci::new(input, output).run()).unwrap();
}

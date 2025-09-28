use anyhow::Error as Failure;
use cinder::{uci::Uci, util::thread};
use clap::Parser;
use futures::{channel::mpsc::unbounded, executor::block_on, sink::unfold as sink};
use std::future::ready;
use std::io::{prelude::*, stdin, stdout};

#[derive(Debug, Parser)]
#[clap(name = "Cinder", version, author)]
#[clap(help_template = "
{name} v{version} by {author}

{name} is a strong chess engine written from scratch.
It is released as free software under the terms of the GNU GPLv3 license.
For more information, visit https://github.com/brunocodutra/cinder#readme.
")]
struct Cli {
    /// Custom hyper-parameters.
    #[cfg(feature = "spsa")]
    #[arg(long)]
    params: Option<cinder::params::Params>,
}

fn main() -> Result<(), Failure> {
    #[allow(unused_variables)]
    let args = Cli::parse();

    #[cfg(feature = "spsa")]
    if let Some(params) = args.params {
        params.init();
    }

    let (tx, input) = unbounded();

    thread::spawn(move || {
        let mut lines = stdin().lock().lines();
        while let Some(Ok(line)) = lines.next() {
            if tx.unbounded_send(line).is_err() {
                break;
            }
        }
    });

    let handle = thread::spawn(move || {
        let mut stdout = stdout().lock();
        let output = sink((), |_, line: String| ready(writeln!(stdout, "{line}")));
        Ok(block_on(Uci::new(input, output).run())?)
    });

    handle.join()
}

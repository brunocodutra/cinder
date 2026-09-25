use anyhow::Error as Failure;
use brotli::Decompressor;
use std::io::{self, Write};
use std::{env, fs::File, path::Path};

fn main() -> Result<(), Failure> {
    let nnue = "lib/nnue/nnue.bin.br";
    println!("cargo:rerun-if-changed={nnue}");
    let compressed = File::open(nnue)?;

    let out_dir = env::var("OUT_DIR")?;
    let dst = Path::new(&out_dir).join("nnue.bin");
    let mut decompressed = File::create(&dst)?;

    let mut decoder = Decompressor::new(compressed, 4096);
    io::copy(&mut decoder, &mut decompressed)?;
    decompressed.flush()?;

    Ok(())
}

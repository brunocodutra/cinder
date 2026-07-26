use anyhow::Error as Failure;
use ruzstd::decoding::StreamingDecoder;
use std::io::{self, Write};
use std::{env, fs::File, path::Path};

fn main() -> Result<(), Failure> {
    let nnue = "lib/nnue/nnue.bin.zst";
    println!("cargo:rerun-if-changed={nnue}");
    let compressed = File::open(nnue)?;

    let out_dir = env::var("OUT_DIR")?;
    let dst = Path::new(&out_dir).join("nnue.bin");
    let mut decompressed = File::create(&dst)?;

    let mut decoder = StreamingDecoder::new_with_max_window_size(compressed, 128 << 20)?;
    io::copy(&mut decoder, &mut decompressed)?;
    decompressed.flush()?;

    Ok(())
}

use crate::util::Assume;
use memmap2::{Mmap, MmapOptions};
use std::{fs::File, io, path::Path, slice::SliceIndex};

#[derive(Debug)]
pub struct RandomAccessFile {
    mmap: Mmap,
}

impl RandomAccessFile {
    /// Opens a file for random read requests.
    #[inline(always)]
    pub fn new(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Random)?;

        Ok(RandomAccessFile { mmap })
    }

    /// Reads the range of bytes specified.
    #[inline(always)]
    pub fn read<I: SliceIndex<[u8]>>(&self, i: I) -> &I::Output {
        self.mmap.get(i).assume()
    }
}

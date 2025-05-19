use memmap2::{Mmap, MmapOptions};
use std::{fs::File, io, path::Path, slice::SliceIndex};

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
    pub fn read<I: SliceIndex<[u8]>>(&self, i: I) -> io::Result<&I::Output> {
        use io::ErrorKind::*;
        self.mmap.get(i).ok_or_else(|| UnexpectedEof.into())
    }

    /// Reads the single byte at a given offset.
    #[inline(always)]
    pub fn read_u8_at(&self, offset: usize) -> io::Result<u8> {
        self.read(offset).copied()
    }

    /// Reads two bytes at a given offset, and interprets them as a
    /// little endian integer.
    #[inline(always)]
    pub fn read_u16_le_at(&self, offset: usize) -> io::Result<u16> {
        let bytes = self.read(offset..offset + 2)?;
        Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
    }
}

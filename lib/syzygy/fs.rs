use crate::util::Assume;
use memmap2::{Mmap, MmapOptions};
use std::{io, path::Path, slice::SliceIndex};

#[derive(Debug)]
pub struct RandomAccessFile {
    mmap: Mmap,
}

impl RandomAccessFile {
    /// Opens a file for random read requests.
    pub fn new(path: &Path) -> io::Result<Self> {
        let mut open_options = std::fs::OpenOptions::new();

        #[cfg(windows)]
        let open_options = {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_RANDOM_ACCESS;
            open_options.custom_flags(FILE_FLAG_RANDOM_ACCESS)
        };

        let file = open_options.read(true).open(path)?;
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

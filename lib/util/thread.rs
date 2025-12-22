use crate::util::Assume;
use std::io;
use std::thread::{Builder, JoinHandle};

/// The stack size for spawned threads.
const STACK_SIZE: usize = 16 << 20;

#[derive(Debug)]
pub struct Handle<T>(JoinHandle<T>);

impl<T> Handle<T> {
    #[track_caller]
    #[inline(always)]
    pub fn join(self) -> T {
        self.0.join().assume()
    }
}

#[track_caller]
#[inline(always)]
pub fn spawn<F, T>(f: F) -> io::Result<Handle<T>>
where
    F: Send + 'static + FnOnce() -> T,
    T: Send + 'static,
{
    Ok(Handle(Builder::new().stack_size(STACK_SIZE).spawn(f)?))
}

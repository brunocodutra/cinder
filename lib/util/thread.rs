use crate::util::Assume;
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
pub fn spawn<F, T>(f: F) -> Handle<T>
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    Handle(Builder::new().stack_size(STACK_SIZE).spawn(f).assume())
}

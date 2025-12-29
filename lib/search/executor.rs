use crate::search::ThreadCount;
use crate::util::{Assume, Int, thread};
use derive_more::with_trait::Debug;
use std::sync::{Arc, Condvar, Mutex};
use std::{cell::SyncUnsafeCell, iter::repeat_n};

struct Barrier {
    lap: u16,
    state: Mutex<(u16, u16)>,
    signal: Condvar,
}

impl Barrier {
    #[inline(always)]
    fn new(lap: u16) -> Self {
        Barrier {
            lap,
            state: Mutex::default(),
            signal: Condvar::default(),
        }
    }

    #[inline(always)]
    fn wait(&self) {
        let mut state = self.state.lock().assume();
        let generation = state.1;
        state.0 += 1;
        if state.0 < self.lap {
            while state.1 == generation {
                state = self.signal.wait(state).assume();
            }
        } else {
            state.0 = 0;
            state.1 = state.1.wrapping_add(1);
            self.signal.notify_all();
        }
    }
}

/// A handle to a task under execution.
#[derive(Debug)]
pub struct Task<'e> {
    executor: &'e mut Executor,
}

impl Drop for Task<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.executor.shared.barrier.wait();
        unsafe { *self.executor.shared.job.get().as_mut_unchecked() = None };
    }
}

type Job = Box<dyn Fn(usize) + Send + Sync>;

#[derive(Debug)]
#[debug("Shared")]
struct Shared {
    barrier: Barrier,
    job: SyncUnsafeCell<Option<Job>>,
}

/// A simple broadcast executor.
#[derive(Debug)]
pub struct Executor {
    shared: Arc<Shared>,
}

impl Drop for Executor {
    #[inline(always)]
    fn drop(&mut self) {
        self.shared.barrier.wait();
    }
}

impl Executor {
    /// Initializes a broadcast executor with the requested number of threads.
    #[inline(always)]
    pub fn new(threads: ThreadCount) -> Self {
        let shared = Arc::new(Shared {
            barrier: Barrier::new(threads.get() + 1),
            job: SyncUnsafeCell::new(None),
        });

        for (idx, shared) in repeat_n(Arc::clone(&shared), threads.cast()).enumerate() {
            thread::spawn(move || {
                loop {
                    shared.barrier.wait();
                    shared.barrier.wait();
                    match unsafe { shared.job.get().as_ref_unchecked() } {
                        Some(job) => job(idx),
                        None => return,
                    }
                }
            });
        }

        shared.barrier.wait();
        Executor { shared }
    }

    /// Executes `f` on every thread.
    #[inline(always)]
    pub fn execute<F: Fn(usize) + Send + Sync + 'static>(&mut self, f: F) -> Task<'_> {
        unsafe { *self.shared.job.get().as_mut_unchecked() = Some(Box::new(f)) };
        self.shared.barrier.wait();
        Task { executor: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
    use std::{fmt::Debug, thread};
    use test_strategy::proptest;

    #[proptest]
    fn barrier_synchronizes_all_threads(c: ThreadCount) {
        let barrier = Barrier::new(c.get() + 1);
        let count = AtomicU16::new(0);

        thread::scope(|s| {
            for _ in 0..c.get() {
                s.spawn(|| {
                    barrier.wait();
                    count.fetch_add(1, Ordering::SeqCst);
                });
            }

            assert_eq!(count.load(Ordering::SeqCst), 0);
            barrier.wait();
        });

        assert_eq!(count.load(Ordering::SeqCst), c.get());
    }

    #[proptest]
    fn executor_runs_job_on_all_threads(c: ThreadCount) {
        let mut executor = Executor::new(c);
        let count = Arc::new(AtomicU16::new(0));

        let task = {
            let count = count.clone();
            executor.execute(move |_| {
                count.fetch_add(1, Ordering::SeqCst);
            })
        };

        drop(task);
        assert_eq!(count.load(Ordering::SeqCst), c.get());
    }

    #[proptest]
    fn executor_runs_exactly_one_job_per_thread(c: ThreadCount) {
        let mut executor = Executor::new(c);
        let count = Arc::new(Vec::from_iter((0..c.get()).map(|_| AtomicUsize::new(0))));

        let task = {
            let count = count.clone();
            executor.execute(move |idx| {
                count[idx].fetch_add(1, Ordering::SeqCst);
            })
        };

        drop(task);

        assert_eq!(
            Vec::from_iter(count.iter().map(|c| c.load(Ordering::SeqCst))),
            vec![1; c.cast()]
        );
    }
}

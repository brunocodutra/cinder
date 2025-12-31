use crate::search::ThreadCount;
use crate::util::{Int, thread};
use derive_more::with_trait::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::{cell::SyncUnsafeCell, iter::repeat_n};

/// A handle to a task under execution.
#[derive(Debug)]
pub struct Execution<'e> {
    executor: &'e mut Executor,
}

impl Drop for Execution<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.executor.shared.barrier.wait();
    }
}

type Task = Box<dyn Fn(usize) + Send + Sync + 'static>;

#[derive(Debug)]
#[debug("Shared")]
struct Shared {
    barrier: Barrier,
    engaged: AtomicUsize,
    task: SyncUnsafeCell<Option<Task>>,
}

/// A simple broadcast executor.
#[derive(Debug)]
pub struct Executor {
    shared: Arc<Shared>,
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.shared.barrier.wait();
    }
}

impl Executor {
    /// Initializes a broadcast executor with the requested number of threads.
    pub fn new(threads: ThreadCount) -> Self {
        let shared = Arc::new(Shared {
            barrier: Barrier::new(1 + threads.cast::<usize>()),
            engaged: AtomicUsize::new(0),
            task: SyncUnsafeCell::new(None),
        });

        for (idx, shared) in repeat_n(Arc::clone(&shared), threads.cast()).enumerate() {
            thread::spawn(move || {
                loop {
                    shared.barrier.wait();
                    shared.engaged.fetch_add(1, Ordering::Relaxed);
                    shared.barrier.wait();

                    match unsafe { shared.task.get().as_ref_unchecked().as_deref() } {
                        Some(task) => task(idx),
                        None => return,
                    }

                    if shared.engaged.fetch_sub(1, Ordering::AcqRel) == 1 {
                        unsafe { drop(shared.task.get().as_mut_unchecked().take()) };
                    }
                }
            });
        }

        shared.barrier.wait();
        Executor { shared }
    }

    /// Executes `f` on every thread.
    #[inline(always)]
    pub fn execute<F: Fn(usize) + Send + Sync + 'static>(&mut self, f: F) -> Execution<'_> {
        let task = unsafe { self.shared.task.get().as_mut_unchecked() };
        task.replace(Box::new(f));
        self.shared.barrier.wait();
        Execution { executor: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use derive_more::with_trait::Deref;
    use std::{fmt::Debug, mem::forget};
    use test_strategy::proptest;

    #[proptest]
    fn executor_runs_task_on_all_threads(c: ThreadCount) {
        let mut executor = Executor::new(c);
        let count = Arc::new(AtomicUsize::new(0));

        let execution = {
            let count = count.clone();
            executor.execute(move |_| {
                count.fetch_add(1, Ordering::SeqCst);
            })
        };

        drop(execution);
        assert_eq!(count.load(Ordering::SeqCst), c.cast::<usize>());
    }

    #[proptest]
    fn executor_runs_task_exactly_once_per_thread(c: ThreadCount) {
        let mut executor = Executor::new(c);
        let count = Arc::new(Vec::from_iter((0..c.get()).map(|_| AtomicUsize::new(0))));

        let execution = {
            let count = count.clone();
            executor.execute(move |idx| {
                count[idx].fetch_add(1, Ordering::SeqCst);
            })
        };

        drop(execution);

        assert_eq!(
            Vec::from_iter(count.iter().map(|c| c.load(Ordering::SeqCst))),
            vec![1; c.cast()]
        );
    }

    #[proptest]
    fn executor_drops_task_once_all_threads_are_done(c: ThreadCount) {
        #[derive(Deref)]
        struct CountDrops(Arc<AtomicUsize>);

        impl Drop for CountDrops {
            fn drop(&mut self) {
                self.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut executor = Executor::new(c);
        let count = Arc::new(AtomicUsize::new(0));
        let drop_counter = CountDrops(count.clone());

        let execution = executor.execute(move |_| {
            assert_eq!(drop_counter.load(Ordering::SeqCst), 0);
        });

        execution.executor.shared.barrier.wait();
        assert_eq!(count.load(Ordering::SeqCst), 1);
        forget(execution);
    }
}

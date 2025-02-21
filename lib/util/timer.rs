use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[ctor::ctor]
pub static EPOCH: Instant = { Instant::now() };

#[inline(always)]
fn elapsed() -> Duration {
    Instant::now().duration_since(*EPOCH)
}

/// Tracks time towards a deadline.
#[derive(Debug, Default)]
pub struct Timer {
    spinner: AtomicU8,
    deadline: Option<Duration>,
}

impl Timer {
    /// Constructs a timer that elapses after the given duration.
    #[inline(always)]
    pub const fn infinite() -> Self {
        Timer {
            spinner: AtomicU8::new(0),
            deadline: None,
        }
    }

    /// Constructs a timer that elapses after the given duration.
    #[inline(always)]
    pub fn new(duration: Duration) -> Self {
        Timer {
            spinner: AtomicU8::new(255),
            deadline: elapsed().checked_add(duration),
        }
    }

    /// Returns the time remaining if any.
    #[inline(always)]
    pub fn remaining(&self) -> Option<Duration> {
        static MICROS: AtomicU64 = AtomicU64::new(0);

        match self.deadline {
            None => Some(Duration::MAX),
            Some(deadline) => {
                if self.spinner.fetch_add(1, Ordering::Relaxed) == 255 {
                    let elapsed = elapsed();
                    MICROS.fetch_max(elapsed.as_micros() as _, Ordering::Relaxed);
                    deadline.checked_sub(elapsed)
                } else {
                    deadline.checked_sub(Duration::from_micros(MICROS.load(Ordering::Relaxed)))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn timer_measures_time_remaining() {
        let timer = Timer::new(Duration::from_secs(1));
        assert!(timer.remaining().is_some());
    }

    #[test]
    fn timer_elapses_once_duration_expires() {
        let timer = Timer::new(Duration::ZERO);
        sleep(Duration::from_millis(1));
        assert_eq!(timer.remaining(), None);
    }

    #[test]
    fn timer_never_decreases_if_infinite() {
        let timer = Timer::infinite();
        let remaining = timer.remaining();
        sleep(Duration::from_millis(1));
        assert_eq!(remaining, timer.remaining());
    }
}

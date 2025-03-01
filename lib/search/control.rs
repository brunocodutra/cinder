use crate::util::{Counter, Timer, Trigger};
use derive_more::with_trait::{Display, Error};

/// Indicates the search was interrupted .
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Error)]
#[display("the search was interrupted")]
pub struct Interrupted;

/// The search control.
#[derive(Debug, Default, Copy, Clone)]
pub enum Control<'a> {
    #[default]
    Unlimited,
    Limited(&'a Counter, &'a Timer, &'a Trigger),
}

impl Control<'_> {
    /// A reference to the node counter.
    #[inline(always)]
    pub fn counter(&self) -> &Counter {
        static INFINITE: Counter = Counter::new(u64::MAX);

        match self {
            Control::Unlimited => &INFINITE,
            Control::Limited(counter, _, _) => counter,
        }
    }

    /// A reference to the timer.
    #[inline(always)]
    pub fn timer(&self) -> &Timer {
        static INFINITE: Timer = Timer::infinite();

        match self {
            Control::Unlimited => &INFINITE,
            Control::Limited(_, timer, _) => timer,
        }
    }

    /// Whether the search should be interrupted.
    #[inline(always)]
    pub fn interrupted(&self) -> Result<(), Interrupted> {
        if let Control::Limited(nodes, timer, trigger) = self {
            nodes.count().ok_or(Interrupted)?;
            timer.remaining().ok_or(Interrupted)?;
            if !trigger.is_armed() {
                return Err(Interrupted);
            }
        }

        Ok(())
    }
}

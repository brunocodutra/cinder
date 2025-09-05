mod bitboard;
mod board;
mod castles;
mod color;
mod file;
mod geometry;
mod magic;
mod r#move;
mod outcome;
mod phase;
mod piece;
mod position;
mod rank;
mod role;
mod square;
mod zobrist;

pub use bitboard::*;
pub use board::*;
pub use castles::*;
pub use color::*;
pub use file::*;
pub use geometry::*;
pub use magic::*;
pub use r#move::*;
pub use outcome::*;
pub use phase::*;
pub use piece::*;
pub use position::*;
pub use rank::*;
pub use role::*;
pub use square::*;
pub use zobrist::*;

/// The butterfly board.
pub type Butterfly<T> = [[T; 64]; 64];

/// The piece-to board.
pub type PieceTo<T> = [[T; 64]; 12];

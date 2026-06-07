mod bitboard;
mod board;
mod castles;
mod color;
mod file;
mod geometry;
mod moves;
mod outcome;
mod phase;
mod piece;
mod pins;
mod placement;
mod position;
mod rank;
mod rays;
mod role;
mod square;
mod threats;
mod zobrist;

pub use bitboard::*;
pub use board::*;
pub use castles::*;
pub use color::*;
pub use file::*;
pub use geometry::*;
pub use moves::*;
pub use outcome::*;
pub use phase::*;
pub use piece::*;
pub use pins::*;
pub use placement::*;
pub use position::*;
pub use rank::*;
pub use rays::*;
pub use role::*;
pub use square::*;
pub use threats::*;
pub use zobrist::*;

/// The butterfly board.
pub type Butterfly<T> = [[T; 64]; 64];

/// The piece-to board.
pub type PieceTo<T> = [[T; 64]; 12];

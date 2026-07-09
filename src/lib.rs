#![allow(dead_code, unused)]
mod parser;
pub mod types;

struct Game {
    white_pieces: u64,
    black_pieces: u64,
    boards: [u64; 6],
}

use crate::{
    features::make_index,
    types::{
        game::{Color, Piece, PieceKind, Position, Square},
        network::FeatureTransformer,
    },
};

const PSQT_BUCKETS: usize = 8;

pub struct Accumulator {
    pub values: Vec<i16>,
    pub psqt_values: [i32; PSQT_BUCKETS],
}

pub struct Accumulators {
    pub white: Accumulator,
    pub black: Accumulator,
}

impl Accumulator {
    pub fn new(half_dims: usize) -> Self {
        Self {
            values: vec![0; half_dims],
            psqt_values: [0; PSQT_BUCKETS],
        }
    }

    pub fn refresh(&mut self, ft: &FeatureTransformer, position: &Position, perspective: Color) {
        self.values.copy_from_slice(&ft.biases);
        self.psqt_values.fill(0);

        let half_dims = self.values.len();
        let king_square = position.king_square(perspective);

        for color in [Color::White, Color::Black] {
            for kind in [
                PieceKind::Pawn,
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
                PieceKind::King,
            ] {
                let piece = Piece { color, kind };
                let mut bitboard = position.pieces(color, kind);

                while bitboard != 0 {
                    let square = bitboard.trailing_zeros() as Square;
                    bitboard &= bitboard - 1;

                    let index = make_index(perspective, square, piece, king_square);
                    let weight_offset = index * half_dims;
                    let psqt_offset = index * PSQT_BUCKETS;

                    for (value, weight) in self
                        .values
                        .iter_mut()
                        .zip(&ft.weights[weight_offset..weight_offset + half_dims])
                    {
                        *value += weight;
                    }
                    for (value, weight) in self
                        .psqt_values
                        .iter_mut()
                        .zip(&ft.psqt_weights[psqt_offset..psqt_offset + PSQT_BUCKETS])
                    {
                        *value += weight;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Accumulator, Color, FeatureTransformer, Piece, PieceKind, Position, make_index};

    #[test]
    fn creates_small_accumulator() {
        let accumulator = Accumulator::new(128);

        assert_eq!(accumulator.values.len(), 128);
        assert_eq!(accumulator.psqt_values, [0; 8]);
    }

    #[test]
    fn creates_big_accumulator() {
        let accumulator = Accumulator::new(1_024);

        assert_eq!(accumulator.values.len(), 1_024);
        assert_eq!(accumulator.psqt_values, [0; 8]);
    }

    #[test]
    fn refreshes_from_biases_and_active_features() {
        let mut accumulator = Accumulator::new(2);
        let position = Position {
            pieces: [[0, 0, 0, 0, 0, 1 << 4], [0, 0, 0, 0, 0, 1 << 60]],
            king_squares: [4, 60],
            side_to_move: Color::White,
        };
        let white_king = Piece {
            color: Color::White,
            kind: PieceKind::King,
        };
        let black_king = Piece {
            color: Color::Black,
            kind: PieceKind::King,
        };
        let white_index = make_index(Color::White, 4, white_king, 4);
        let black_index = make_index(Color::White, 60, black_king, 4);
        let mut weights = vec![0; 22_528 * 2];
        let mut psqt_weights = vec![0; 22_528 * 8];

        weights[white_index * 2..white_index * 2 + 2].copy_from_slice(&[1, 2]);
        weights[black_index * 2..black_index * 2 + 2].copy_from_slice(&[3, 4]);
        psqt_weights[white_index * 8..white_index * 8 + 8].fill(1);
        psqt_weights[black_index * 8..black_index * 8 + 8].fill(2);

        let ft = FeatureTransformer {
            biases: vec![5, 6],
            weights,
            psqt_weights,
            threat_weights: None,
            threat_psqt_weights: None,
        };

        accumulator.refresh(&ft, &position, Color::White);

        assert_eq!(accumulator.values, [9, 12]);
        assert_eq!(accumulator.psqt_values, [3; 8]);
    }
}

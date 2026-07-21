use crate::types::game::{Color, Piece, PieceKind, Square};

const SQUARES: usize = 64;
const PIECE_CATEGORIES: usize = 11;
const KING_BUCKET_SIZE: usize = SQUARES * PIECE_CATEGORIES;
pub const HALF_KA_INPUT_DIMS: usize = 32 * KING_BUCKET_SIZE;

// Rows are ranks 1 through 8; columns are files a through h.
#[rustfmt::skip]
const ORIENT_TBL: [usize; SQUARES] = [
    7, 7, 7, 7, 0, 0, 0, 0, // rank 1
    7, 7, 7, 7, 0, 0, 0, 0, // rank 2
    7, 7, 7, 7, 0, 0, 0, 0, // rank 3
    7, 7, 7, 7, 0, 0, 0, 0, // rank 4
    7, 7, 7, 7, 0, 0, 0, 0, // rank 5
    7, 7, 7, 7, 0, 0, 0, 0, // rank 6
    7, 7, 7, 7, 0, 0, 0, 0, // rank 7
    7, 7, 7, 7, 0, 0, 0, 0, // rank 8
 ];

// Rows are ranks 1 through 8; columns are files a through h.
#[rustfmt::skip]
const KING_BUCKETS: [usize; SQUARES] = [
    28, 29, 30, 31, 31, 30, 29, 28, // rank 1
    24, 25, 26, 27, 27, 26, 25, 24, // rank 2
    20, 21, 22, 23, 23, 22, 21, 20, // rank 3
    16, 17, 18, 19, 19, 18, 17, 16, // rank 4
    12, 13, 14, 15, 15, 14, 13, 12, // rank 5
     8,  9, 10, 11, 11, 10,  9,  8, // rank 6
     4,  5,  6,  7,  7,  6,  5,  4, // rank 7
     0,  1,  2,  3,  3,  2,  1,  0, // rank 8
 ];

pub fn make_index(perspective: Color, square: Square, piece: Piece, king_square: Square) -> usize {
    let perspective = perspective as usize;
    let flip = 56 * perspective;
    let king_square = king_square as usize;

    let oriented_square = square as usize ^ ORIENT_TBL[king_square] ^ flip;
    let king_bucket = KING_BUCKETS[king_square ^ flip] * KING_BUCKET_SIZE;

    king_bucket + piece_square_index(perspective, piece) + oriented_square
}

fn piece_square_index(perspective: usize, piece: Piece) -> usize {
    if piece.kind == PieceKind::King {
        return 10 * SQUARES;
    }

    let relative_color = piece.color as usize ^ perspective;
    (piece.kind as usize * 2 + relative_color) * SQUARES
}

#[cfg(test)]
mod tests {
    use super::{HALF_KA_INPUT_DIMS, make_index};
    use crate::types::game::{Color, Piece, PieceKind};

    const WHITE_PAWN: Piece = Piece {
        color: Color::White,
        kind: PieceKind::Pawn,
    };
    const BLACK_PAWN: Piece = Piece {
        color: Color::Black,
        kind: PieceKind::Pawn,
    };

    #[test]
    fn white_perspective_preserves_right_side_king_positions() {
        assert_eq!(make_index(Color::White, 12, WHITE_PAWN, 4), 21_836);
    }

    #[test]
    fn white_perspective_horizontally_mirrors_left_side_king_positions() {
        assert_eq!(make_index(Color::White, 15, WHITE_PAWN, 3), 21_832);
    }

    #[test]
    fn black_perspective_vertically_mirrors_the_board() {
        assert_eq!(make_index(Color::Black, 52, BLACK_PAWN, 60), 21_836);
    }

    #[test]
    fn black_perspective_can_apply_both_mirrors() {
        assert_eq!(make_index(Color::Black, 55, BLACK_PAWN, 59), 21_832);
    }

    #[test]
    fn both_kings_use_the_shared_king_category() {
        let white_king = Piece {
            color: Color::White,
            kind: PieceKind::King,
        };
        let black_king = Piece {
            color: Color::Black,
            kind: PieceKind::King,
        };

        assert_eq!(
            make_index(Color::White, 4, white_king, 4),
            make_index(Color::White, 4, black_king, 4)
        );
    }

    #[test]
    fn indices_stay_within_the_feature_transformer() {
        for perspective in [Color::White, Color::Black] {
            for king_square in 0..64 {
                for square in 0..64 {
                    for piece in [WHITE_PAWN, BLACK_PAWN] {
                        assert!(
                            make_index(perspective, square, piece, king_square)
                                < HALF_KA_INPUT_DIMS
                        );
                    }
                }
            }
        }
    }
}

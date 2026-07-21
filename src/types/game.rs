pub type Square = u8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Color {
    White,
    Black,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PieceKind {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub color: Color,
    pub kind: PieceKind,
}

pub struct Position {
    pub pieces: [[u64; 6]; 2],
    pub king_squares: [Square; 2],
    pub side_to_move: Color,
}

impl Position {
    pub fn pieces(&self, color: Color, kind: PieceKind) -> u64 {
        self.pieces[color as usize][kind as usize]
    }

    pub fn occupancy(&self, color: Color) -> u64 {
        let [pawns, knights, bishops, rooks, queens, king] = self.pieces[color as usize];

        pawns | knights | bishops | rooks | queens | king
    }

    pub fn piece_at(&self, square: Square) -> Option<Piece> {
        if square >= 64 {
            return None;
        }

        let square_bit = 1_u64 << square;
        for color in [Color::White, Color::Black] {
            for kind in [
                PieceKind::Pawn,
                PieceKind::Knight,
                PieceKind::Bishop,
                PieceKind::Rook,
                PieceKind::Queen,
                PieceKind::King,
            ] {
                if self.pieces(color, kind) & square_bit != 0 {
                    return Some(Piece { color, kind });
                }
            }
        }

        None
    }

    pub fn king_square(&self, color: Color) -> Square {
        self.king_squares[color as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::{Color, Piece, PieceKind, Position};

    fn position() -> Position {
        Position {
            pieces: [[1 << 8, 0, 0, 0, 0, 1 << 4], [0, 1 << 57, 0, 0, 0, 1 << 60]],
            king_squares: [4, 60],
            side_to_move: Color::White,
        }
    }

    #[test]
    fn accesses_piece_bitboards() {
        let position = position();
        assert_eq!(position.pieces(Color::White, PieceKind::Pawn), 1 << 8);
        assert_eq!(position.pieces(Color::Black, PieceKind::Knight), 1 << 57);
    }

    #[test]
    fn combines_color_occupancy() {
        let position = position();
        assert_eq!(position.occupancy(Color::White), (1 << 4) | (1 << 8));
        assert_eq!(position.occupancy(Color::Black), (1 << 57) | (1 << 60));
    }

    #[test]
    fn finds_piece_at_square() {
        let position = position();
        assert_eq!(
            position.piece_at(57),
            Some(Piece {
                color: Color::Black,
                kind: PieceKind::Knight,
            })
        );
        assert_eq!(position.piece_at(9), None);
        assert_eq!(position.piece_at(64), None);
    }

    #[test]
    fn accesses_king_squares() {
        let position = position();
        assert_eq!(position.king_square(Color::White), 4);
        assert_eq!(position.king_square(Color::Black), 60);
    }
}

use super::utils::notation_to_index;
use std::fmt::{self, Display};

#[derive(Default)]
pub struct Bboard(u64);

impl Bboard {
    pub fn new() -> Self {
        Self(0)
    }

    pub fn bits(self) -> u64 {
        self.0
    }

    pub fn set_bits(&mut self, squares: &str) -> Result<(), String> {
        for square in squares
            .split(',')
            .map(str::trim)
            .filter(|square| !square.is_empty())
        {
            let index =
                notation_to_index(square).ok_or_else(|| format!("invalid square: {square}"))?;
            self.0 |= 1u64 << index;
        }

        Ok(())
    }
}

impl Display for Bboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const RESET: &str = "\x1b[0m";
        const MUTED: &str = "\x1b[38;5;244m";
        const POP: &str = "\x1b[1;38;5;220m";
        const LABEL: &str = MUTED;

        writeln!(f)?;

        for rank in (0..8).rev() {
            write!(f, " {LABEL}{}{RESET}   ", rank + 1)?;

            for file in 0..8 {
                let index = rank * 8 + file;
                if self.0 & (1u64 << index) != 0 {
                    write!(f, "{POP}1{RESET}  ")?;
                } else {
                    write!(f, "{MUTED}.{RESET}  ")?;
                }
            }

            writeln!(f, " {LABEL}{}{RESET}", rank + 1)?;
        }
        writeln!(f, "     {LABEL}a  b  c  d  e  f  g  h{RESET}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_bits_sets_requested_squares() {
        let mut board = Bboard::new();
        board.set_bits("a2,a4,b4").unwrap();

        assert_eq!(board.bits(), (1u64 << 8) | (1u64 << 24) | (1u64 << 25));
    }

    #[test]
    fn set_bits_rejects_invalid_squares() {
        let mut board = Bboard::new();

        assert_eq!(board.set_bits("z9").unwrap_err(), "invalid square: z9");
    }
    #[test]
    fn visual_prints_board() {
        let mut board = Bboard::new();
        board.set_bits("a1,h8,d5").unwrap();
        println!("{}", board);
    }
}

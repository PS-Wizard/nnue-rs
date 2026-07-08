pub fn notation_to_index(square: &str) -> Option<u8> {
    if square.len() != 2 {
        return None;
    }

    let bytes = square.as_bytes();
    let file = bytes[0].to_ascii_lowercase();
    let rank = bytes[1];

    if !(b'a'..=b'h').contains(&file) || !(b'1'..=b'8').contains(&rank) {
        return None;
    }

    Some((rank - b'1') * 8 + (file - b'a'))
}

#[cfg(test)]
mod tests {
    use super::notation_to_index;

    #[test]
    fn notation_to_index_maps_board_positions() {
        assert_eq!(notation_to_index("a1"), Some(0));
        assert_eq!(notation_to_index("h1"), Some(7));
        assert_eq!(notation_to_index("a8"), Some(56));
        assert_eq!(notation_to_index("h8"), Some(63));
        assert_eq!(notation_to_index("a2"), Some(8));
    }

    #[test]
    fn notation_to_index_rejects_invalid_input() {
        assert_eq!(notation_to_index("i4"), None);
        assert_eq!(notation_to_index("a9"), None);
        assert_eq!(notation_to_index("aa1"), None);
    }
}

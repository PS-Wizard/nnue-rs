use anyhow::{Context, Result};

pub struct NNUEReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> NNUEReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        NNUEReader { bytes, offset: 0 }
    }

    pub fn finish(&self) -> Result<()> {
        anyhow::ensure!(
            self.offset == self.bytes.len(),
            "{} trailing unread bytes",
            self.bytes.len() - self.offset
        );
        Ok(())
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        let bytes: [u8; 4] = self.read_bytes(4)?.try_into().unwrap();
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn read_bytes(&mut self, len: usize) -> Result<&[u8]> {
        let start = self.offset;
        let end = start
            .checked_add(len)
            .context("failed to read bytes: offset + length overflow")?;
        let slice = self
            .bytes
            .get(start..end)
            .context("unexpected EOF; failed to read bytes")?;

        self.offset = end;
        Ok(slice)
    }

    pub fn read_leb128_i16(&mut self, count: usize) -> Result<Vec<i16>> {
        let payload = self.read_leb128_payload()?;
        let mut offset = 0;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(
                i16::try_from(read_signed_leb128(payload, &mut offset)?)
                    .context("decoded LEB128 value is outside i16 range")?,
            );
        }
        ensure_payload_consumed(payload, offset).context("failed to decode i16 LEB128 values")?;
        Ok(values)
    }

    pub fn read_leb128_i32(&mut self, count: usize) -> Result<Vec<i32>> {
        let payload = self.read_leb128_payload()?;
        let mut offset = 0;
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(read_signed_leb128(payload, &mut offset)?);
        }
        ensure_payload_consumed(payload, offset).context("failed to decode i32 LEB128 values")?;
        Ok(values)
    }

    /// Reads one shared LEB128 block and splits it into two i32 arrays.
    pub fn read_shared_leb128_i32(
        &mut self,
        first_count: usize,
        second_count: usize,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        let payload = self.read_leb128_payload()?;
        let mut offset = 0;
        let mut first = Vec::with_capacity(first_count);
        for _ in 0..first_count {
            first.push(read_signed_leb128(payload, &mut offset)?);
        }
        let mut second = Vec::with_capacity(second_count);
        for _ in 0..second_count {
            second.push(read_signed_leb128(payload, &mut offset)?);
        }
        ensure_payload_consumed(payload, offset)
            .context("failed to decode shared i32 LEB128 values")?;
        Ok((first, second))
    }

    fn read_leb128_payload(&mut self) -> Result<&[u8]> {
        const MAGIC: &[u8] = b"COMPRESSED_LEB128";
        let actual_magic = self
            .read_bytes(MAGIC.len())
            .context("failed to read LEB128 block magic")?;
        anyhow::ensure!(actual_magic == MAGIC, "invalid LEB128 block magic");

        let byte_count = self
            .read_u32()
            .context("failed to read LEB128 payload length")? as usize;
        self.read_bytes(byte_count)
            .context("failed to read LEB128 payload")
    }

    pub fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_bytes(1)?[0] as i8)
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        let bytes: [u8; 4] = self.read_bytes(4)?.try_into().unwrap();
        Ok(i32::from_le_bytes(bytes))
    }

    pub fn read_i8_vec(&mut self, count: usize) -> Result<Vec<i8>> {
        (0..count).map(|_| self.read_i8()).collect()
    }

    pub fn read_i32_vec(&mut self, count: usize) -> Result<Vec<i32>> {
        (0..count).map(|_| self.read_i32()).collect()
    }
}

fn ensure_payload_consumed(payload: &[u8], offset: usize) -> Result<()> {
    anyhow::ensure!(
        offset == payload.len(),
        "LEB128 payload has {} leftover bytes",
        payload.len() - offset
    );
    Ok(())
}

fn read_signed_leb128(payload: &[u8], offset: &mut usize) -> Result<i32> {
    let mut result = 0u32;
    let mut shift = 0usize;
    for byte_index in 0..5 {
        let byte = *payload
            .get(*offset)
            .context("unexpected EOF while decoding signed LEB128 value")?;
        *offset += 1;
        result |= ((byte & 0x7f) as u32) << (shift % 32);

        if byte & 0x80 == 0 {
            if byte_index == 4 && !matches!(byte & 0x7f, 0x00..=0x07 | 0x78..=0x7f) {
                anyhow::bail!("malformed signed i32 LEB128 value");
            }
            if byte & 0x40 != 0 {
                let sign_shift = shift + 7;
                if sign_shift < 32 {
                    result |= !((1u32 << sign_shift) - 1);
                }
            }
            return Ok(result as i32);
        }
        shift += 7;
    }
    anyhow::bail!("malformed signed i32 LEB128 value: continuation exceeds five bytes")
}

#[cfg(test)]
mod tests {
    use super::NNUEReader;

    fn block(payload: &[u8]) -> Vec<u8> {
        let mut bytes = b"COMPRESSED_LEB128".to_vec();
        bytes.extend((payload.len() as u32).to_le_bytes());
        bytes.extend(payload);
        bytes
    }

    #[test]
    fn reads_positive_and_negative_values() {
        let bytes = block(&[0x05, 0x7b, 0xc7, 0x01, 0xb9, 0x7e]);
        let mut reader = NNUEReader::new(&bytes);
        assert_eq!(reader.read_leb128_i32(4).unwrap(), vec![5, -5, 199, -199]);
    }

    #[test]
    fn reads_shared_i32_block() {
        let bytes = block(&[0x01, 0x7e, 0x03]);
        let mut reader = NNUEReader::new(&bytes);
        assert_eq!(
            reader.read_shared_leb128_i32(2, 1).unwrap(),
            (vec![1, -2], vec![3])
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = block(&[0]);
        bytes[0] = b'X';
        let mut reader = NNUEReader::new(&bytes);
        assert!(reader.read_leb128_i32(1).is_err());
    }

    #[test]
    fn rejects_leftover_payload() {
        let bytes = block(&[1, 2]);
        let mut reader = NNUEReader::new(&bytes);
        assert!(reader.read_leb128_i16(1).is_err());
    }

    #[test]
    fn rejects_malformed_overlong_leb128() {
        let bytes = block(&[0x80, 0x80, 0x80, 0x80, 0x80]);
        let mut reader = NNUEReader::new(&bytes);
        assert!(reader.read_leb128_i32(1).is_err());
    }

    #[test]
    fn decodes_i32_max_sleb128() {
        // i32::MAX = 0x7FFFFFFF → 5-byte SLEB128: FF FF FF FF 07
        let bytes = block(&[0xff, 0xff, 0xff, 0xff, 0x07]);
        let mut reader = NNUEReader::new(&bytes);
        assert_eq!(reader.read_leb128_i32(1).unwrap(), vec![i32::MAX]);
    }

    #[test]
    fn decodes_i32_min_sleb128() {
        // i32::MIN = -2147483648 → 5-byte SLEB128: 80 80 80 80 78
        let bytes = block(&[0x80, 0x80, 0x80, 0x80, 0x78]);
        let mut reader = NNUEReader::new(&bytes);
        assert_eq!(reader.read_leb128_i32(1).unwrap(), vec![i32::MIN]);
    }

    #[test]
    fn rejects_eof_while_reading_block() {
        let bytes = b"COMPRESSED_LEB128";
        let mut reader = NNUEReader::new(bytes);
        assert!(reader.read_leb128_i32(0).is_err());
    }

    #[test]
    fn finish_rejects_trailing_bytes() {
        let bytes = [1, 2, 3];
        let reader = NNUEReader::new(&bytes);
        assert!(reader.finish().is_err());
    }

    #[test]
    fn finish_accepts_consumed_bytes() {
        let bytes = [1, 2, 3];
        let mut reader = NNUEReader::new(&bytes);
        reader.read_bytes(3).unwrap();
        assert!(reader.finish().is_ok());
    }
}

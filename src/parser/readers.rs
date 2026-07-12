use anyhow::{Context, Result, anyhow};

pub struct NNUEReader<'a> {
    bytes: &'a Vec<u8>,
    offset: usize,
}

impl<'a> NNUEReader<'a> {
    pub fn new(bytes: &'a Vec<u8>) -> Self {
        NNUEReader { bytes, offset: 0 }
    }

    pub fn read_u32(&mut self) -> Result<u32> {
        let slice = self
            .bytes
            .get(self.offset..self.offset + 4)
            .context("unexpected EOF; failed to read u32")?;

        self.offset += 4;
        Ok(u32::from_le_bytes(slice.try_into().unwrap()))
    }

    pub fn read_bytes(&mut self, len: usize) -> Result<&[u8]> {
        let start = self.offset;
        let end = start + len;

        let slice = self
            .bytes
            .get(start..end)
            .context("unexpected EOF; failed to read bytes")?;

        self.offset = end;
        Ok(slice)
    }
}

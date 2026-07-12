use std::{
    fs::File,
    io::{BufReader, Read},
};

use crate::{parser::readers::NNUEReader, types::network::NetworkSpec};
use anyhow::Context;

mod dense_network;
mod feature_transformer;
mod header;
mod readers;

pub fn read_nnue(spec: NetworkSpec) -> anyhow::Result<Vec<u8>> {
    let file = File::open(spec.path)?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;

    let mut nnreader = NNUEReader::new(&bytes);

    let magic = nnreader.read_u32().context("failed to read magic")?;
    println!("magic: {:#010X}", magic);

    let hash = nnreader.read_u32().context("failed to read hash")?;
    println!("hash: {:#010X}", hash);

    let len_description = nnreader
        .read_u32()
        .context("failed to read description length")?;
    println!("description_len: {}", len_description);

    let description = nnreader
        .read_bytes(len_description as usize)
        .context("failed to read description")?;

    let description = String::from_utf8_lossy(description);
    println!("description: {}", description);

    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use crate::{
        parser::read_nnue,
        types::network::{NetworkKind, NetworkSpec},
    };

    #[test]
    fn test_read_nnue() {
        let path = "./networks/halfka_v2/nn-37f18f62d772.nnue";
        let network_kind = NetworkKind::Small(path.to_string());
        let spec = NetworkSpec::new(network_kind);
        let bytes = read_nnue(spec).unwrap();
        let metadata = std::fs::metadata(path).unwrap();
        println!("File size: {} bytes", metadata.len());
        println!("Bytes read: {}", bytes.len());
    }
}

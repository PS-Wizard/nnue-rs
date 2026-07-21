use std::{
    fs::File,
    io::{BufReader, Read},
};

use crate::{
    parser::reader::NNUEReader,
    types::network::{
        DenseLayer, FeatureTransformer, LayerStack, Network, NetworkHeader, NetworkKind,
        NetworkSpec,
    },
};
use anyhow::Context;

mod reader;

pub fn read_nnue(spec: NetworkSpec) -> anyhow::Result<Network> {
    let file = File::open(&spec.path)?;
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    let mut nnreader = NNUEReader::new(&bytes);

    let version = nnreader.read_u32().context("failed to read magic")?;
    println!("magic: {:#010X}", version);
    anyhow::ensure!(
        version == 0x7AF32F20,
        "unexpected NNUE version: expected {:#010X}, got {version:#010X}",
        0x7AF32F20,
    );

    let network_hash = nnreader.read_u32().context("failed to read hash")?;
    println!("hash: {:#010X}", network_hash);
    let expected_file_hash = spec.transformer_hash ^ spec.architecture_hash;
    anyhow::ensure!(
        network_hash == expected_file_hash,
        "unexpected network hash: expected {expected_file_hash:#010X}, got {network_hash:#010X}",
    );

    let len_description = nnreader
        .read_u32()
        .context("failed to read description length")?;
    println!("description_len: {}", len_description);

    let description = nnreader
        .read_bytes(len_description as usize)
        .context("failed to read description")?;

    let description = String::from_utf8_lossy(description);
    println!("description: {}", description);

    let network_header = NetworkHeader {
        version,
        network_hash,
        description: description.to_string(),
    };

    let ft_hash = nnreader.read_u32().context("failed to read ft hash")?;
    println!("ft_hash: {:#010X}", ft_hash);
    anyhow::ensure!(
        ft_hash == spec.transformer_hash,
        "unexpected feature-transformer hash: expected {:#010X}, got {ft_hash:#010X}",
        spec.transformer_hash,
    );

    let (kind, transformer) = match spec.threat_input_dims {
        None => {
            let biases = nnreader.read_leb128_i16(spec.half_dims)?;
            println!("Biases Lenght: {}", biases.len());

            let weights = nnreader.read_leb128_i16(spec.half_dims * spec.input_dims)?;
            println!("Weights Lenght: {}", weights.len());

            let psqt_weights = nnreader.read_leb128_i32(spec.input_dims * 8)?;
            println!("PSQT weights length: {}", psqt_weights.len());

            (
                NetworkKind::Small(spec.path),
                FeatureTransformer {
                    biases,
                    weights,
                    psqt_weights,
                    threat_weights: None,
                    threat_psqt_weights: None,
                },
            )
        }
        Some(threat_input_dims) => {
            let biases = nnreader.read_leb128_i16(spec.half_dims)?;
            println!("Biases Lenght: {}", biases.len());

            let threat_weights = nnreader.read_i8_vec(threat_input_dims * spec.half_dims)?;
            println!("Weights Lenght: {}", threat_weights.len());

            let weights = nnreader.read_leb128_i16(spec.input_dims * spec.half_dims)?;
            println!("Weights Lenght: {}", weights.len());

            let (threat_psqt_weights, psqt_weights) =
                nnreader.read_shared_leb128_i32(threat_input_dims * 8, spec.input_dims * 8)?;
            println!("Threat PSQT weights length: {}", threat_psqt_weights.len());
            println!("PSQT weights length: {}", psqt_weights.len());

            (
                NetworkKind::Big(spec.path),
                FeatureTransformer {
                    biases,
                    weights,
                    psqt_weights,
                    threat_weights: Some(threat_weights),
                    threat_psqt_weights: Some(threat_psqt_weights),
                },
            )
        }
    };

    let mut layer_stacks = Vec::with_capacity(8);

    for _ in 0..8 {
        let hash = nnreader.read_u32()?;
        anyhow::ensure!(
            hash == spec.architecture_hash,
            "unexpected layer-stack hash: {hash:#010X}"
        );

        let fc0 = DenseLayer {
            biases: nnreader.read_i32_vec(16)?,
            weights: nnreader.read_i8_vec(16 * spec.half_dims)?,
        };

        let fc1 = DenseLayer {
            biases: nnreader.read_i32_vec(32)?,
            weights: nnreader.read_i8_vec(32 * 32)?,
        };

        let fc2 = DenseLayer {
            biases: nnreader.read_i32_vec(1)?,
            weights: nnreader.read_i8_vec(32)?,
        };

        layer_stacks.push(LayerStack { fc0, fc1, fc2 });
    }

    nnreader
        .finish()
        .context("unexpected trailing bytes after network")?;

    Ok(Network {
        header: network_header,
        kind,
        transformer,
        layer_stacks,
    })
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
        let network = read_nnue(spec).unwrap();
        let metadata = std::fs::metadata(path).unwrap();
        println!("File size: {} bytes", metadata.len());
        println!("Layer stacks read: {}", network.layer_stacks.len());
    }

    #[test]
    fn test_read_big_nnue() {
        let path = "./networks/full_threats/nn-c288c895ea92.nnue";
        let spec = NetworkSpec::new(NetworkKind::Big(path.to_string()));
        let network = read_nnue(spec).unwrap();

        assert_eq!(network.layer_stacks.len(), 8);
        assert_eq!(network.transformer.biases.len(), 1_024);
        assert_eq!(network.transformer.weights.len(), 22_528 * 1_024);
        assert_eq!(network.transformer.psqt_weights.len(), 22_528 * 8);
        assert_eq!(
            network.transformer.threat_weights.as_ref().unwrap().len(),
            79_856 * 1_024
        );
        assert_eq!(
            network
                .transformer
                .threat_psqt_weights
                .as_ref()
                .unwrap()
                .len(),
            79_856 * 8
        );
    }
}

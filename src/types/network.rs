pub struct NetworkSpec {
    pub path: String,
    pub transformer_hash: u32,
    pub architecture_hash: u32,
    pub half_dims: usize,
    pub input_dims: usize,
    pub threat_input_dims: Option<usize>,
}

pub enum NetworkKind {
    Small(String),
    Big(String),
}

impl NetworkSpec {
    pub fn new(kind: NetworkKind) -> NetworkSpec {
        match kind {
            NetworkKind::Small(path) => NetworkSpec {
                path,
                transformer_hash: 0x7f234db8,
                architecture_hash: 0x6333712a,
                half_dims: 128,
                input_dims: 22_528,
                threat_input_dims: None,
            },
            NetworkKind::Big(path) => NetworkSpec {
                path,
                transformer_hash: 0x8f2344b8,
                architecture_hash: 0x63336a4a,
                half_dims: 1_024,
                input_dims: 22_528,
                threat_input_dims: Some(79_856),
            },
        }
    }
}

pub struct NetworkHeader {
    pub version: u32,
    pub network_hash: u32,
    pub description: String,
}

pub struct FeatureTransformer {
    pub biases: Vec<i16>,
    pub weights: Vec<i16>,
    pub psqt_weights: Vec<i32>,

    // only used by big network
    pub threat_weights: Option<Vec<i8>>,
    pub threat_psqt_weights: Option<Vec<i32>>,
}

pub struct LayerStack {
    pub fc0: DenseLayer,
    pub fc1: DenseLayer,
    pub fc2: DenseLayer,
}

pub struct DenseLayer {
    pub biases: Vec<i32>,
    pub weights: Vec<i8>,
}

pub struct Network {
    pub header: NetworkHeader,
    pub kind: NetworkKind,
    pub transformer: FeatureTransformer,
    pub layer_stacks: Vec<LayerStack>,
}

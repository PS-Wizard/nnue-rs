# Layer Stacks and Forward Pass

> After the feature transformer, the network is 8 independent small MLPs ("layer stacks"). One is selected per position by piece count. This note gives the exact forward pass with dimensions, types, and the skip connection.

Source: `nnue/nnue_architecture.h` (`NetworkArchitecture::propagate`), `nnue/layers/*.h`.

## Bucket selection

```
bucket = (popcount(all pieces) - 1) / 4        // 0..7
```

32 pieces → bucket 7, ≤ 4 pieces → bucket 0. Each bucket's stack was trained on positions of that piece-count range (endgames get different weights than middlegames). The same `bucket` also selects the PSQT column ([[07 - Feature Transformer and Accumulator]]).

## Architecture of one stack

Constants: `L2 = 15`, `L3 = 32` (identical for big and small nets; only L1 differs).

```
input: u8[L1]                       (from transform())
  │
fc_0   AffineTransformSparseInput<L1, 16>      → i32[16]
  │        16 = L2+1: outputs 0..14 are hidden neurons,
  │        output 15 is a SKIP CONNECTION straight to the end
  ├────────────────────────────┐
ac_sqr_0  SqrClippedReLU<16>   │   → u8, squared-clip of fc_0 outs
ac_0      ClippedReLU<16>      │   → u8, linear-clip of fc_0 outs
  │                            │
  │  concat: buffer[0..15]  = ac_sqr_0 out (only 0..14 meaningful)
  │          buffer[15..30] = ac_0 out[0..14]   (memcpy over index 15!)
  ▼                            │
fc_1   AffineTransform<30, 32>  │  → i32[32]     (padded input dim 32)
ac_1   ClippedReLU<32>          │  → u8[32]
fc_2   AffineTransform<32, 1>   │  → i32[1]
  │                            │
  ▼                            ▼
positional = fc_2out[0] + fc_0out[15] * (600*16) / (127*64)
```

### The dual activation (ac_sqr_0 ++ ac_0)

fc_1's 30 inputs are **two views of the same 15 fc_0 neurons**:
- inputs 0..14: `SqrClippedReLU` — `min(127, (x*x) >> 19)`
- inputs 15..29: `ClippedReLU` — `clamp(x >> 6, 0, 127)`

Implementation detail from Stockfish: it runs ac_sqr_0 over all 16 values into the buffer, then memcpys ac_0's first 15 values to `buffer + 15`, overwriting the squared value of the skip output at index 15. Inputs 30/31 (padding) are zero. Your implementation can simply compute the 30 values directly.

### The skip connection ("fwdOut")

fc_0's 16th output is a trained direct-eval term. Scale conversion (derivation in [[03 - Data Types and Quantization]]):

```
positional = fc_2_out[0] + fc_0_out[15] * 9600 / 8128
```

(9600 = 600·OutputScale, 8128 = 127·2^WeightScaleBits. Integer division, truncating.)

## Layer semantics (scalar reference)

### AffineTransform / AffineTransformSparseInput

Mathematically identical; "sparse input" is a runtime optimization exploiting that most u8 inputs are 0 after the FT activation (see [[11 - SIMD Optimizations]]). Reference:

```
out[o] = bias[o] + Σ_i  (weight[o][i] as i32) * (input[i] as i32)
         // weight: i8, input: u8, out: i32
```

Weight storage in file: row-major `[out][padded_in]`. In memory Stockfish scrambles it — [[12 - Weight Permutation and Scrambling]] — but a scalar Rust port can keep row-major.

Padding: `PaddedInputDimensions = ceil_to_multiple(in, 32)`. fc_1's file weights include the two zero padding columns (32 stored per row for 30 real inputs); fc_0's L1 is already a multiple of 32.

### ClippedReLU

```
out[i] = clamp(in[i] >> 6, 0, 127) as u8
```

### SqrClippedReLU

```
out[i] = min(127, ((in[i] as i64) * in[i]) >> 19) as u8
```

## Network output

`Network::evaluate` returns the pair (both already divided by `OutputScale = 16`):

```
psqt_value       = transform_psqt / 16    // "material-ish" half
positional_value = propagate(...) / 16    // "positional" half
```

These two halves are blended (not just summed!) by the caller — see [[10 - Big-Small Net Switching and Final Eval]].

## Sizes cheat sheet (per stack)

| layer | weights | biases | in → out types |
|---|---|---|---|
| fc_0 | i8 × 16×L1 | i32 × 16 | u8[L1] → i32[16] |
| fc_1 | i8 × 32×32 | i32 × 32 | u8[30 (+2 pad)] → i32[32] |
| fc_2 | i8 × 1×32 | i32 × 1 | u8[32] → i32[1] |

8 stacks per net; each preceded in the file by its hash u32 — [[01 - NNUE File Format]].

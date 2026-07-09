# NNUE File Format (SF 18)

> Byte-exact layout of both `.nnue` files. Everything below was verified by parsing the real files to EOF with an independent Python script — every count and hash matched.

All multi-byte integers are **little-endian**. There is no alignment or padding in the file itself (padding exists only in memory).

## Top-level structure

```
┌────────────────────────────────────────────┐
│ File header                                │
├────────────────────────────────────────────┤
│ u32 hash  +  FeatureTransformer parameters │
├────────────────────────────────────────────┤
│ u32 hash  +  LayerStack[0] parameters      │
│ u32 hash  +  LayerStack[1] parameters      │
│ ...              (8 stacks total)          │
│ u32 hash  +  LayerStack[7] parameters      │
└────────────────────────────────────────────┘  ← must be exactly EOF
```

Stockfish validates `stream.peek() == EOF` after the last stack — your parser should too.

## File header

| offset | size | field | value |
|---|---|---|---|
| 0 | 4 | `version` | `0x7AF32F20` (u32) |
| 4 | 4 | `hash` | big net: `0xEC102EF2`, small net: `0x1C103C92` |
| 8 | 4 | `desc_len` | length of description string |
| 12 | desc_len | `description` | UTF-8, no NUL. Current nets: "Network trained with the https://github.com/official-stockfish/nnue-pytorch trainer." (84 bytes) |

The `hash` is `FeatureTransformer::get_hash_value() ^ NetworkArchitecture::get_hash_value()` — a *structure* hash (depends only on architecture, not weights). Full derivation in [[14 - Constants Reference]]; short version:

```
ft_hash   = FeatureSetHash ^ (L1 * 2)
            FullThreats:  0x8F234CB8 ^ 2048 = 0x8F2344B8   (big)
            HalfKAv2_hm:  0x7F234CB8 ^  256 = 0x7F234DB8   (small)
arch_hash = chain over layers, see [[14 - Constants Reference]]
            big: 0x63336A4A     small: 0x6333712A
file hash = ft_hash ^ arch_hash
```

## Feature transformer section

Starts with `u32 = ft_hash` (values above), then the parameters. **The layout differs between the two nets.**

Dimensions used below:

| symbol | value | meaning |
|---|---|---|
| `L1` | 1024 (big) / 128 (small) | accumulator half-width |
| `HKA` | 22528 | HalfKAv2_hm feature count |
| `THREAT` | 79856 | FullThreats feature count |
| `PSQTB` | 8 | PSQT buckets |

### Big net (FullThreats + HalfKAv2_hm)

In file order:

| # | content | encoding | element type | count |
|---|---|---|---|---|
| 1 | `biases` | [[02 - LEB128 Compression\|LEB128 block]] | i16 | 1024 |
| 2 | `threatWeights` | **raw** little-endian bytes (NOT LEB!) | **i8** | 79856 × 1024 = 81,772,544 |
| 3 | `weights` (HalfKA) | LEB128 block | i16 | 22528 × 1024 = 23,068,672 |
| 4 | `threatPsqtWeights` ++ `psqtWeights` | **one single LEB128 block** containing both arrays back-to-back | i32 | 79856×8 = 638,848 then 22528×8 = 180,224 (819,072 total) |

Gotchas:
- Section 2 is plain bytes — one i8 per weight, 81.7 MB. This is why the big net file is ~109 MB.
- Section 4 is *one* LEB block (one magic string, one byte-count) whose decoded stream fills `threatPsqtWeights` first, then `psqtWeights`. Stockfish does `read_leb_128(stream, threatPsqtWeights, psqtWeights)` — the variadic call shares a single compressed region.

Weight matrix orientation (all FT weights): **feature-major / column-major per feature**. `weights[feature * L1 + j]` is the contribution of `feature` to accumulator lane `j`. Same for `threatWeights`. PSQT: `psqtWeights[feature * 8 + bucket]`.

### Small net (HalfKAv2_hm only)

| # | content | encoding | element type | count |
|---|---|---|---|---|
| 1 | `biases` | LEB128 block | i16 | 128 |
| 2 | `weights` | LEB128 block | i16 | 22528 × 128 = 2,883,584 |
| 3 | `psqtWeights` | LEB128 block | i32 | 22528 × 8 = 180,224 |

After reading, the small net's `weights` and `biases` (not psqt) are **multiplied by 2** in memory (`scale_weights(true)`) — a SIMD activation trick, see [[11 - SIMD Optimizations]]. When Stockfish *exports* a net it divides by 2 again. Threat nets are not scaled.

Also after reading, both nets' FT arrays get the **packus permutation** applied in memory ([[12 - Weight Permutation and Scrambling]]). This does not affect the file.

## Layer stack sections (× 8)

Each stack starts with `u32 = arch_hash` (`0x63336A4A` big / `0x6333712A` small), then in order: `fc_0`, `fc_1`, `fc_2`. The ReLU layers (`ac_0`, `ac_1`, `ac_sqr_0`) have **no parameters** (their `read_parameters` is a no-op) but they DO participate in the hash chain.

All layer parameters are **raw little-endian**, no LEB128.

### fc_0 — AffineTransformSparseInput<L1, 16>

| content | type | count | notes |
|---|---|---|---|
| biases | i32 | 16 | |
| weights | i8 | 16 × L1 | file order: row-major `[output][input]`, `PaddedInputDimensions = L1` (1024 and 128 are already multiples of 32, so no file padding) |

Output 16 = `L2 + 1` = 15 real neurons + 1 **skip connection** output (see [[08 - Layer Stacks and Forward Pass]]).

### fc_1 — AffineTransform<30, 32>

| content | type | count | notes |
|---|---|---|---|
| biases | i32 | 32 | |
| weights | i8 | 32 × 32 = 1024 | `InputDimensions = 30` (2×15), but `PaddedInputDimensions = 32` — the file **contains the padding columns** (they are zero) |

### fc_2 — AffineTransform<32, 1>

| content | type | count |
|---|---|---|
| biases | i32 | 1 |
| weights | i8 | 1 × 32 = 32 |

Per-stack payload: `64 + 16·L1 + 128 + 1024 + 4 + 32` bytes (plus the 4-byte hash).

In-memory only: dense-layer weights are stored **scrambled** for the SIMD kernels (`get_weight_index_scrambled`); the file order is the natural row-major order. See [[12 - Weight Permutation and Scrambling]].

## Loading algorithm (what your Rust loader must do)

1. Read header; check `version == 0x7AF32F20` and `hash` matches the expected structure hash for the net you're loading.
2. Read `u32`; check it equals `ft_hash`. Read FT parameters per the table above.
3. For `i in 0..8`: read `u32`; check `== arch_hash`; read fc_0/fc_1/fc_2.
4. Verify EOF.
5. (Optional, needed only for SIMD parity) apply packus permutation + ×2 scaling (small net) + dense-weight scrambling.

Related: [[02 - LEB128 Compression]], [[03 - Data Types and Quantization]], [[14 - Constants Reference]].

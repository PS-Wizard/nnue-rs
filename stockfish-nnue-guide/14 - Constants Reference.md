# Constants Reference

> Every magic number in one place. All values verified against SF 18 source and the real network files.

## File-level

| constant | value |
|---|---|
| version | `0x7AF32F20` |
| LEB magic | `"COMPRESSED_LEB128"` (17 bytes, no NUL) |
| default big net | `nn-c288c895ea92.nnue` (~109 MB) |
| default small net | `nn-37f18f62d772.nnue` (~1.2 MB) |
| download | `https://tests.stockfishchess.org/api/nn/<name>` |

## Architecture dimensions

| | big | small |
|---|---|---|
| L1 (`TransformedFeatureDimensions`) | 1024 | 128 |
| L2 (`FC_0_OUTPUTS`) | 15 | 15 |
| L3 (`FC_1_OUTPUTS`) | 32 | 32 |
| feature sets | FullThreats + HalfKAv2_hm | HalfKAv2_hm |
| `PSQTBuckets` | 8 | 8 |
| `LayerStacks` | 8 | 8 |
| HalfKAv2_hm dims | 22,528 | 22,528 |
| FullThreats dims | 79,856 | — |
| max active features | 32 (PSQ), 128 (threats) | 32 |

## Hashes (all verified against real files)

Structure-hash algorithm:

```
u32 wrapping arithmetic throughout.

feature_transformer_hash = FeatureSetHash ^ (L1 * 2)
affine_hash(prev, outDims) = { h = 0xCC03DAE4 + outDims; h ^= prev >> 1; h ^= prev << 31; h }
crelu_hash(prev)           = 0x538D24C7 + prev

arch_hash(L1, L2, L3):
    h = 0xEC42E90D ^ (L1 * 2)
    h = affine_hash(h, L2 + 1)   // fc_0
    h = crelu_hash(h)            // ac_0   (ac_sqr_0 NOT in the chain)
    h = affine_hash(h, L3)       // fc_1
    h = crelu_hash(h)            // ac_1
    h = affine_hash(h, 1)        // fc_2

file_hash = feature_transformer_hash ^ arch_hash
```

| hash | big | small |
|---|---|---|
| feature set (`HashValue`) | FullThreats `0x8F234CB8` | HalfKAv2_hm `0x7F234CB8` |
| FT section hash | `0x8F2344B8` | `0x7F234DB8` |
| layer-stack section hash | `0x63336A4A` | `0x6333712A` |
| file header hash | `0xEC102EF2` | `0x1C103C92` |
| affine layer id | `0xCC03DAE4` | |
| ReLU layer id | `0x538D24C7` | |
| input-slice id | `0xEC42E90D` | |

## Quantization / eval constants

| constant | value | where |
|---|---|---|
| `OutputScale` | 16 | network output → Value |
| `WeightScaleBits` | 6 | ClippedReLU shift |
| skip-connection scale | `× (600·16) / (127·64)` = ×9600/8128 | [[08 - Layer Stacks and Forward Pass]] |
| pairwise activation divisor | 512 | [[07 - Feature Transformer and Accumulator]] |
| SqrClippedReLU shift | 19 (= 2·6+7) | |
| clamp max (big / small) | 255 / 254 | |
| bucket formula | `(pieceCount − 1) / 4` | |

## evaluate.cpp constants (SF 18 — retuned every release)

| constant | value |
|---|---|
| small-net threshold | \|simple_eval\| > 962 |
| big-net re-eval threshold | \|nnue\| < 277 |
| psqt/positional blend | (125·psqt + 131·positional) / 128 |
| optimism complexity | / 476 |
| nnue complexity damp | / 18236 |
| material term | 534·pawns + non_pawn_material (both sides) |
| final blend | (nnue·(77871+mat) + optimism·(7191+mat)) / 77871 |
| rule-50 damp | v·rule50 / 199 |

## Piece values (`types.h`)

| piece | value |
|---|---|
| Pawn | 208 |
| Knight | 781 |
| Bishop | 825 |
| Rook | 1276 |
| Queen | 2538 |

## Piece / square encodings

```
Color: WHITE=0, BLACK=1
PieceType: PAWN=1 ... KING=6
Piece = color*8 + type:  W_PAWN=1..W_KING=6, B_PAWN=9..B_KING=14
Square = rank*8 + file:  A1=0, H1=7, A8=56, H8=63, SQ_NONE=64
flip file: sq ^ 7   flip rank: sq ^ 56   flip piece color: pc ^ 8
```

## FullThreats layout table (verified)

| attacker | base | pairs/slot | slots (`numValidTargets`) |
|---|---|---|---|
| W_PAWN | 0 | 84 | 6 |
| W_KNIGHT | 504 | 336 | 12 |
| W_BISHOP | 4536 | 560 | 10 |
| W_ROOK | 10136 | 896 | 10 |
| W_QUEEN | 19096 | 1456 | 12 |
| W_KING | 36568 | 420 | 8 |
| B_PAWN | 39928 | 84 | 6 |
| B_KNIGHT | 40432 | 336 | 12 |
| B_BISHOP | 44464 | 560 | 10 |
| B_ROOK | 50064 | 896 | 10 |
| B_QUEEN | 59024 | 1456 | 12 |
| B_KING | 76496 | 420 | 8 |
| total | **79856** | | |

## DirtyThreat bit layout (u32)

| bits | field |
|---|---|
| 0–7 | attacker square |
| 8–15 | victim square |
| 16–19 | victim piece |
| 20–23 | attacker piece |
| 31 | add flag |

List capacity 96 (≤ 80 real + 16 SIMD slack).

## SIMD parameters

| ISA | `vec_t` | regs | `MaxChunkSize` | packus order |
|---|---|---|---|---|
| AVX-512 | 512b | 16 | 64 | 0,2,4,6,1,3,5,7 |
| AVX2 | 256b | 12 | 32 | 0,2,1,3,4,6,5,7 |
| SSE2 | 128b | 12 (64-bit) | 16 | identity |
| NEON | 128b | 16 | 16 | identity |

Index: [[00 - Stockfish NNUE Overview]]

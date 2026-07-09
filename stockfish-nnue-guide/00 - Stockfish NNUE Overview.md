# Stockfish NNUE Overview (SF 18, FullThreats era)

> The big picture: what NNUE is, how all the pieces fit together, and where to read next.
> Everything in this guide was verified byte-for-byte against the actual SF 18 source and the real network files (`nn-c288c895ea92.nnue` big, `nn-37f18f62d772.nnue` small).

## What NNUE is

NNUE = **Efficiently Updatable Neural Network**. It is a small quantized (integer-only) neural network that evaluates a chess position in the order of a few thousand CPU instructions. Two properties make it fast:

1. **The first layer is enormous but sparse.** Inputs are binary features ("white knight on f3, king on g1" or "knight on f3 attacks pawn on e5"). Only ~30–130 of tens of thousands of features are active at once. The first layer's output — the **accumulator** — is just the sum of the weight columns for active features plus a bias.
2. **Moves change few features.** A move flips a handful of features on/off, so instead of recomputing the accumulator you add/subtract a few weight columns. That's the "efficiently updatable" part. See [[09 - Incremental Updates]].

Everything after the accumulator is tiny (a few 30→32→1 dense layers) and is recomputed from scratch every evaluation.

## The two networks

SF 18 carries **two independent networks** and picks per-position (see [[10 - Big-Small Net Switching and Final Eval]]):

| | Big net | Small net |
|---|---|---|
| default file | `nn-c288c895ea92.nnue` (~109 MB) | `nn-37f18f62d772.nnue` (~1.2 MB) |
| accumulator width (L1) | 1024 | 128 |
| feature sets | [[04 - HalfKAv2_hm Feature Set\|HalfKAv2_hm]] **+** [[05 - FullThreats Feature Set\|FullThreats]] | HalfKAv2_hm only |
| used when | position is balanced (needs accuracy) | material is lopsided (\|simple_eval\| > 962) |

Both use the identical downstream architecture (`L2 = 15`, `L3 = 32`, 8 layer stacks, 8 PSQT buckets) — only L1 and the feature sets differ.

## The evaluation pipeline, end to end

```
Position
  │
  ├─ features: HalfKAv2_hm (piece-square, king-relative)   → active indices
  ├─ features: FullThreats (attacker→victim threats, big net only)
  │        indices computed by make_index  ──────────── [[06 - make_index Deep Dive]]
  │
  ▼
Feature Transformer  (the huge sparse first layer) ───── [[07 - Feature Transformer and Accumulator]]
  accumulator[side][1024] : i16       (sum of weight columns + bias, updated incrementally)
  psqtAccumulation[side][8] : i32     (a scalar material-ish value per bucket, same trick)
  │
  ▼ transform(): clip + pairwise multiply halves → u8[1024], pick PSQT bucket
  │
  ▼
Layer stack [bucket = (pieceCount-1)/4, 8 of them] ───── [[08 - Layer Stacks and Forward Pass]]
  fc_0: 1024 → 16 (sparse input)   [15 neurons + 1 skip-connection output]
  ac_sqr_0 / ac_0: squared-clipped and clipped ReLU, concatenated → 30
  fc_1: 30 → 32,  ac_1: clip,  fc_2: 32 → 1
  │
  ▼
(psqt, positional) value pair
  │
  ▼
Eval::evaluate(): blend psqt+positional, complexity damping,
material scaling, optimism, 50-move damping ──────────── [[10 - Big-Small Net Switching and Final Eval]]
```

## How the file on disk maps to that pipeline

The `.nnue` file (see [[01 - NNUE File Format]]) is exactly: a header, the feature-transformer weights (mostly [[02 - LEB128 Compression|LEB128-compressed]]), then 8 copies of the layer-stack weights (raw little-endian). All datatypes and quantization scales are in [[03 - Data Types and Quantization]].

At **load time** Stockfish additionally rearranges weights in memory for SIMD:
- a **packus permutation** of the feature-transformer columns ([[12 - Weight Permutation and Scrambling]]),
- a **4-byte scramble** of the dense-layer weights (same note),
- for the small net only, weights/biases are **multiplied by 2** (a trick enabling a `mulhi`-based activation, see [[11 - SIMD Optimizations]]).

A crucial consequence for your Rust port: **the file format is SIMD-agnostic**; all the weird orderings happen after reading. You can write a dead-simple scalar parser first, verify it, and add the permutations only when you write the SIMD kernels.

## Search-time machinery

During search Stockfish keeps an **AccumulatorStack** — one entry per ply, each holding the big + small accumulators and a diff record of what the move changed (`DirtyPiece` for piece-square features, `DirtyThreats` for threat features). Accumulators are computed lazily: when an evaluation is requested, the stack walks back to the last computed entry and applies diffs forward (or rebuilds from scratch/cache when a king moved across a refresh boundary). Full-refresh cost is mitigated by **Finny tables** (per-king-square cached accumulators). All in [[09 - Incremental Updates]].

## Reading order

Beginner → advanced:

1. [[01 - NNUE File Format]] — the bytes on disk
2. [[02 - LEB128 Compression]] — decode the compressed blocks
3. [[03 - Data Types and Quantization]] — what the numbers mean
4. [[04 - HalfKAv2_hm Feature Set]] and [[05 - FullThreats Feature Set]] — what the inputs are
5. [[06 - make_index Deep Dive]] — features → weight-array indices
6. [[07 - Feature Transformer and Accumulator]] — the first layer
7. [[08 - Layer Stacks and Forward Pass]] — the rest of the network
8. [[10 - Big-Small Net Switching and Final Eval]] — producing the final score
9. [[09 - Incremental Updates]] — making it fast across moves
10. [[11 - SIMD Optimizations]] and [[12 - Weight Permutation and Scrambling]] — making it *really* fast
11. [[13 - Rust Implementation Roadmap]] — your project plan
12. [[14 - Constants Reference]] — every magic number in one place

## Source files (in `reference/Stockfish-sf_18/src/`)

| File | What it holds |
|---|---|
| `nnue/nnue_common.h` | types, version, LEB128, endian readers |
| `nnue/nnue_architecture.h` | dims (L1/L2/L3), layer-stack definition, propagate() |
| `nnue/network.h/.cpp` | file header, load/save, evaluate entry point |
| `nnue/nnue_feature_transformer.h` | FT weights layout, transform(), permutations |
| `nnue/features/half_ka_v2_hm.h/.cpp` | PSQ feature set + its make_index |
| `nnue/features/full_threats.h/.cpp` | threat feature set + its make_index |
| `nnue/layers/*.h` | affine layers, ReLUs, weight scrambling |
| `nnue/nnue_accumulator.h/.cpp` | accumulator stack, incremental updates, Finny cache |
| `nnue/simd.h` | all SIMD abstractions and tiling |
| `evaluate.cpp` | net selection + final blend |
| `types.h` | `DirtyPiece`, `DirtyThreats`, piece encodings |
| `position.h/.cpp` | threat-diff generation during do_move |

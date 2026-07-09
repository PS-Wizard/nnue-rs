# Feature Transformer and Accumulator

> The feature transformer (FT) is the giant sparse first layer. Its "forward pass" is split in two: the **accumulator** (maintained incrementally across moves) and **transform()** (a cheap per-eval activation step producing the u8 input for the layer stacks).

Source: `nnue/nnue_feature_transformer.h`, `nnue/nnue_accumulator.h`.

## Parameters (per net)

```
biases            : [i16; L1]                       // L1 = 1024 big / 128 small
weights           : [i16; 22528 * L1]               // HalfKAv2_hm columns
threatWeights     : [i8 ; 79856 * L1]               // big net only
psqtWeights       : [i32; 22528 * 8]
threatPsqtWeights : [i32; 79856 * 8]                // big net only
```

Column layout: `weights[f * L1 + j]` — each feature owns a contiguous L1-wide column. This is the key to fast accumulation: adding a feature = one contiguous vector add.

## The accumulator

```
struct Accumulator<L1> {
    accumulation:     [[i16; L1]; 2],   // [perspective][lane]
    psqtAccumulation: [[i32; 8]; 2],    // [perspective][bucket]
    computed:         [bool; 2],
}
```

Definition (per perspective):

```
accumulation[p][j]   = biases[j] + Σ_{f ∈ activePSQ(p)} weights[f*L1 + j]
                                 (+ nothing from threats — see below)
psqtAccumulation[p][b] = Σ_{f ∈ activePSQ(p)} psqtWeights[f*8 + b]
```

**Big net twist:** the two feature sets keep **separate accumulator states** (`psq_accumulators` and `threat_accumulators` in the `AccumulatorStack`) because their diffs and refresh rules differ; they are summed only inside `transform()`:

```
threatAcc[p][j]   = Σ_{f ∈ activeThreats(p)} threatWeights[f*L1 + j]   // no bias
threatPsqt[p][b]  = Σ_{f ∈ activeThreats(p)} threatPsqtWeights[f*8 + b]
```

Note: threat accumulation lanes are i16 even though weights are i8 (widened when added). The threat accumulator has **no bias** — the bias lives once, in the PSQ accumulator.

How accumulators are computed lazily and updated move-to-move: [[09 - Incremental Updates]].

## transform() — accumulator → u8 features

Called once per evaluation with the selected PSQT/layer-stack `bucket` (see [[08 - Layer Stacks and Forward Pass]] for bucket selection). Steps:

### 1. PSQT output (the "material" half of the eval)

```
psqt = psqAcc.psqt[us][bucket] - psqAcc.psqt[them][bucket]
big net:  psqt = (psqt + threatAcc.psqt[us][bucket] - threatAcc.psqt[them][bucket]) / 2
small net: psqt = psqt / 2
```

`us` = side to move. This i32 is returned from `transform()` and becomes the `psqt` half of the network output (divided by `OutputScale` later).

### 2. Pairwise-multiply activation → output buffer

The output is `L1` u8 values: first the side-to-move perspective's half, then the opponent's (`offset = (L1/2) * p` with the halves-multiply — see below). For each perspective `p` (0 = stm, 1 = opponent), scalar reference semantics:

```
for j in 0 .. L1/2:
    sum0 = acc[persp[p]][j]              // first half lane
    sum1 = acc[persp[p]][j + L1/2]       // second half lane

    big net:  sum0 = clamp(sum0 + threatAcc[persp[p]][j],        0, 255)
              sum1 = clamp(sum1 + threatAcc[persp[p]][j + L1/2], 0, 255)
    small:    sum0 = clamp(sum0, 0, 254)     // 254 = 127*2 (weights were doubled at load)
              sum1 = clamp(sum1, 0, 254)

    output[p * L1/2 + j] = (sum0 * sum1) / 512     // u8, ≤ 127 (255·255/512)
```

So each perspective contributes `L1/2` outputs (pairs of lanes multiplied together — "squared" activation), total buffer = `L1` u8. Side to move always occupies the first half — this is how the net knows whose turn it is.

Why `/512` and the 254 clamp: [[03 - Data Types and Quantization]]. The SIMD version replaces clamp+multiply with a min/shift/`mulhi`/`packus` sequence that needs the weights pre-permuted: [[11 - SIMD Optimizations]] and [[12 - Weight Permutation and Scrambling]].

## The Finny-table cache (`AccumulatorCaches::Cache`)

Full refreshes of the PSQ accumulator (needed on own-king moves) don't start from zero. A per-thread cache stores, for **each (king square, perspective)** pair:

```
struct Entry {
    accumulation:     [i16; L1],     // accumulator for the cached board
    psqtAccumulation: [i32; 8],
    pieces:           [Piece; 64],   // the board that accumulator describes
    pieceBB:          Bitboard,
}
```

A "refresh" then diffs the *current* board against the cached board (usually just a few pieces differ), applies add/sub columns, stores the result back to the cache **and** the accumulator. Entries are initialized to `accumulation = biases`, empty board. Details in [[09 - Incremental Updates]].

FullThreats has **no cache** — its refresh (`update_threats_accumulator_full`) rebuilds from the active-feature list (≤128 column adds). Threat refreshes are rare (king crossing the d/e boundary only).

## Buffer sizes and alignment

- FT output buffer: `L1` bytes, 64-byte aligned (`CacheLineSize`).
- All weight arrays 64-byte aligned. In Rust: allocate with an aligned wrapper (e.g. `#[repr(C, align(64))] struct Aligned<T>(T)` around boxed slices, or a small aligned-vec helper). Alignment matters once you use aligned SIMD loads.

Next: [[08 - Layer Stacks and Forward Pass]].

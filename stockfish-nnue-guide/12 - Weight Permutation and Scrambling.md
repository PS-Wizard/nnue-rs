# Weight Permutation and Scrambling

> Two distinct load-time weight reorderings ("the weird orders"). Both exist so that SIMD kernels can use pack/dot instructions whose lane semantics would otherwise deliver results in the wrong order. **The file always stores the natural order**; these transforms are memory-only. A scalar implementation can skip both entirely.

## 1. The packus permutation (feature transformer)

### The problem

`_mm256_packus_epi16(v0, v1)` doesn't concatenate its inputs; it interleaves them per 128-bit lane:

```
AVX2 result lanes (64-bit blocks):  v0[0..1], v1[0..1], v0[2..3], v1[2..3]
AVX-512:                            interleaves 128-bit lanes of v0/v1 similarly
```

In the FT activation ([[11 - SIMD Optimizations]]), pairs of i16 accumulator vectors are multiplied and packed to u8. Without correction, output bytes would land permuted relative to fc_0's weight rows.

### The fix: permute the *weights* once at load

Reorder the accumulator lanes themselves (by permuting FT weight/bias columns) with the inverse interleave, so that after packus everything lands in natural order. The permutation acts on **groups of 64 lanes**, viewed as 8 blocks of 8 lanes, reordered by:

| ISA | `PackusEpi16Order` |
|---|---|
| AVX-512 | `[0, 2, 4, 6, 1, 3, 5, 7]` |
| AVX2 | `[0, 2, 1, 3, 4, 6, 5, 7]` |
| SSE2 / NEON / scalar | identity (packus already concatenates at 128-bit width) |

Concretely (`permute_weights` / `permute<BlockSize>`):

- `biases` (i16): permute blocks of 16 bytes (8 lanes) within every 128-byte (64-lane) window.
- `weights` (i16): same — each feature's L1-wide column is a multiple of 64 lanes, so the pattern repeats down the whole array.
- `threatWeights` (i8): blocks of **8 bytes** (still 8 lanes) within every 64-byte window — element-wise identical permutation.

PSQT weights are untouched (they never go through packus).

Pseudocode:

```
for every consecutive window of 64 elements:
    new[block j] = old[block order[j]]     // blocks of 8 elements
```

`unpermute_weights` (the inverse) is applied before saving a net. If you write your Rust engine with scalar or 128-bit code only, use the identity order and your accumulator memory layout matches the file directly.

**Important interaction:** the permutation must be consistent between the FT (which produces the u8 buffer) and fc_0's weights — but fc_0 consumes the buffer through the *sparse* path where input chunk index `i` addresses weight block `i`; since the FT output is permuted, and fc_0 weights are stored by input index in file order... Stockfish resolves this by permuting the FT so the *output ends up in file-natural order after packus*. So fc_0 sees natural order and needs no matching change. Your invariant to test: `transform()` SIMD output == scalar `transform()` output, byte for byte.

## 2. The 4-byte scramble (dense affine layers)

### The problem

The `dpbusd` kernel ([[11 - SIMD Optimizations]]) broadcasts 4 consecutive input bytes and needs, adjacent in memory: **those 4 input positions' weights for every output**. The natural row-major layout `[out][in]` scatters them.

### The layout

`get_weight_index_scrambled(i)` for logical flat index `i` in `[out][padded_in]` order:

```
scrambled = (i / 4) % (PaddedIn / 4) * OutDims * 4    // which input-chunk block
          + (i / PaddedIn) * 4                        // which output within the block
          + (i % 4)                                   // byte within the chunk
```

Equivalently: the weight for (output `o`, input `k`) lives at
```
block = k / 4, byte = k % 4
mem[block * (OutDims*4) + o * 4 + byte]
```
i.e. a `[in_chunks][out][4]` layout. The kernel then reads `&weights[chunk * OutDims * 4]` as `NumRegs` consecutive vectors covering all outputs.

Applied on **read** (file order → scrambled memory) for both `AffineTransform` and `AffineTransformSparseInput`, only when an SSSE3/NEON-dotprod-class kernel is compiled (`ENABLE_SEQ_OPT`); the non-SIMD fallback keeps row-major. fc_2 (OutDims = 1) uses the scramble formula too — for OutDims=1 it degenerates to... `(i/4)*4 + i%4 = i`: identity, conveniently.

### Rust guidance

Keep the *file-order* row-major weights as the canonical representation and derive the scrambled buffer when constructing the SIMD evaluator:

```rust
for i in 0..(OUT * PADDED_IN) {
    scrambled[scrambled_index(i)] = file_order[i];
}
```

Then your scalar evaluator (row-major) and SIMD evaluator (scrambled) can coexist and cross-check.

## Summary table

| transform | applies to | when | granularity | purpose |
|---|---|---|---|---|
| packus permutation | FT biases, weights, threatWeights | load (and inverse on save) | 8-element blocks in 64-element windows | fix packus lane interleave in transform() |
| ×2 scaling | small-net FT weights+biases | load (÷2 on save) | scalar | enable mulhi shift trick, see [[03 - Data Types and Quantization]] |
| 4-byte scramble | all affine layer weights | load | `[in_chunk][out][4]` | contiguous vectors for dpbusd broadcast kernel |

Related: [[11 - SIMD Optimizations]], [[01 - NNUE File Format]].

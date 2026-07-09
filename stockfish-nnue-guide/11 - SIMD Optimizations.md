# SIMD Optimizations

> Where NNUE's speed actually comes from. This note explains each SIMD technique conceptually and with enough precision to reproduce it in Rust (via `std::arch` intrinsics or `std::simd`). The load-time weight reorderings that these kernels rely on are in [[12 - Weight Permutation and Scrambling]].

Stockfish supports AVX-512(+VNNI/ICL), AVX2, SSSE3/SSE4.1, SSE2, and NEON(+dotprod). Vector width: `vec_t` = 512/256/128 bits. A scalar fallback exists for every path — **your scalar Rust implementation is exactly that fallback, and it's the ground truth to test SIMD against.**

## 1. Accumulator updates: tiled add/sub of weight columns

Adding/removing a feature = adding/subtracting an L1-wide i16 column ([[09 - Incremental Updates]]). The tiling scheme (`SIMDTiling` in `simd.h`):

- Split the L1 lanes into **tiles** that fit the register file: `TileHeight = NumRegs * lanes_per_reg`. E.g. AVX2: 12 usable regs... actually `NumRegistersSIMD` = 16 (AVX-512), 12 (AVX2), 12/6 (SSE2 64/32-bit), 16 (NEON); `BestRegisterCount` picks the largest count whose tile divides L1.
- For each tile: load accumulator tile → registers, loop over all added/removed features adding/subtracting their column slices, store once.

This turns k feature-updates into one read-modify-write of the accumulator instead of k.

PSQT columns (8 × i32) get the same treatment with `psqt_vec_t` (256-bit AVX2 → whole PSQT row in one register).

**Threat columns are i8**: they're widened while streaming — `vec_convert_8_16` (`_mm256_cvtepi8_epi16` etc.), NEON uses `vmovl_s8` low/high halves. Note the load reads a *half-width* vector of i8 (`vec_i8_t`) and widens to a full `vec_t`.

The multi-operand HalfKA update is a **fused chain**: `out = in + colA - colR0 - colR1` evaluated per register — one loop, no intermediate stores (`fused_row_reduce`, variadic `fused<Add, Sub, Sub>`).

## 2. transform(): the packus/mulhi activation

Scalar goal per output byte ([[07 - Feature Transformer and Accumulator]]): `clamp₀..MAX(a) * clampₘᵢₙ..MAX(b) / 512 → u8`.

The SIMD trick chain (from the long comment in `nnue_feature_transformer.h`):

1. **Implicit low clamp via packus**: `packus_epi16` saturates negative i16 to 0 when packing to u8. If `b` is negative, `a*b` is negative, and packus zeroes it — same result as clamping `b` to 0 first. Saves one `max` per pair: only `a` needs the full `[0,MAX]` clamp, `b` only needs `min(b, MAX)`.
2. **Signed multiply with built-in shift via mulhi**: `mulhi_epi16` returns the high 16 bits = `(a*b) >> 16`, preserving sign. To net `/512` (shift 9): pre-shift `a` left by 7 (`slli_epi16`), then `mulhi` ⇒ `>> 16` ⇒ net `>> 9`. The `[0,255]`/`[0,254]` clamp guarantees `a << 7` fits in i16 without touching the sign bit (255·128 = 32640 < 32767). This is *why* the small net's weights are doubled at load: it moves the required total shift to 9 so that the pre-shift of 7 stays safe.
   - NEON: `vqdmulhq_s16` doubles the product, so pre-shift by **6** instead of 7.
3. **packus lane problem** ⇒ solved at load time by permuting weight columns ([[12 - Weight Permutation and Scrambling]]): `_mm256_packus_epi16(v0, v1)` interleaves 64-bit chunks of its inputs rather than concatenating them; pre-permuted weights make the packed bytes land in natural feature order.

Big-net variant: before the clamp, add the threat accumulator tile (`vec_add_16(in, tin)`), clamp to 255. Small net: clamp to 254.

Per output chunk (pseudocode, one perspective):

```
acc0 = in0[2j],  acc0b = in0[2j+1]        // first-half lanes (a)
acc1 = in1[2j],  acc1b = in1[2j+1]        // second-half lanes (b)
sum0  = slli(clamp(acc0, 0, MAX), 7)      // full clamp + preshift
sum1  = min(acc1, MAX)                    // upper clamp only
p     = mulhi(sum0, sum1)
out[j] = packus(p, p_b)
```

## 3. Dense affine layers: dpbusd and the 4-byte weight scramble

Core primitive `vec_add_dpbusd_32(acc, a, b)`: multiply 4 unsigned bytes of `a` with 4 signed bytes of `b` pairwise and accumulate the 4 products into one i32 lane — that's AVX-VNNI's `_mm256_dpbusd_epi32`, emulated on SSSE3/AVX2 with `maddubs + madd(1)`, NEON `vdotq_s32`.

To use it, the input is viewed as i32 chunks (4 bytes each) and broadcast: `in = set1_epi32(input32[i])`; weights are stored so that the 4 input bytes × all outputs form contiguous vectors (the scramble — [[12 - Weight Permutation and Scrambling]]). Then per input chunk: one broadcast + `NumRegs` dpbusd ops accumulate into all outputs simultaneously.

- fc_1 (30→32): loops all 8 input chunks.
- fc_2 (32→1): different shape — single output, so it's a dot product: `hadd(Σ dpbusd(in_vec, weight_vec))`.

## 4. Sparse input for fc_0 (`AffineTransformSparseInput`)

After the squared activation most FT outputs are **zero** (typically ~75–90%). fc_0 exploits this:

1. **find_nnz**: scan the u8[L1] input as i32 chunks (4 bytes), building the list of indices of nonzero 4-byte chunks. Vectorized: compare-gt-zero → movemask → for each 8-bit mask group, look up a precomputed table `offset_indices[256][8]` of set-bit positions and store `base + offsets` with one unaligned 128-bit store; `count += popcount(mask)`. (AVX-512 uses compress instructions instead.)
2. Loop only over nonzero chunks: broadcast the 4-byte chunk, dpbusd against the 16-output weight block for that chunk.

With VNNI, accumulators are split into 3 dependency chains (dpbusd has multi-cycle latency) processing 3 chunks per iteration, merged at the end.

Rust note: the `offset_indices` LUT approach works great with `std::arch`; a simpler first version can just do `while mask != 0 { idx = mask.trailing_zeros(); ... mask &= mask-1 }`.

## 5. ClippedReLU vectorization

i32 → u8 with `>> 6` clamp: pack pairs of i32 vectors to i16 with `packus_epi32`/`packs_epi32`, shift right 6, pack again to i8 with `packs_epi16`, fix lane order with a final 32-bit permute (`_mm256_permutevar8x32_epi32` with offsets `{0,4,1,5,2,6,3,7}`) — needed for the same packus-interleave reason as in the FT (here fixed at runtime because these buffers are transient). SqrClippedReLU similarly uses `mulhi(x,x) >> 3` (16-bit squares of packed i16, total effective shift 19).

## 6. Threat-diff generation with AVX-512 (optional flourish)

On AVX512-ICL builds, `Position::update_piece_threats` writes up to 16 `DirtyThreat` records in one shot: byte-compress the square list from a bitboard mask, gather victim pieces with a byte permute of the 64-byte board array, then OR square/piece fields into a broadcast template and store 512 bits (`write_multiple_dirties`). The 96-entry list's 16 slack slots absorb the unmasked store. Purely optional for a port — the scalar loop is equivalent.

## 7. Miscellany

- `get_changed_pieces` (Finny refresh): compare two 64-byte board arrays with two 256-bit `cmpeq_epi8` + movemask → changed-square bitboard.
- All hot arrays are 64-byte aligned; loads/stores use aligned variants where possible.
- Buffers between layers are per-eval stack buffers (Stockfish uses a thread_local; in Rust just put them in a reusable struct — no allocation in the eval path).

Related: [[12 - Weight Permutation and Scrambling]], [[13 - Rust Implementation Roadmap]].

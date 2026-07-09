# Data Types and Quantization

> NNUE inference is pure integer arithmetic. This note lists every datatype in the pipeline and the fixed-point scales that make the math work.

## The type aliases (nnue_common.h)

| Stockfish alias | Rust type | used for |
|---|---|---|
| `BiasType` | `i16` | feature-transformer biases (and accumulator lanes) |
| `WeightType` | `i16` | feature-transformer HalfKA weights |
| `ThreatWeightType` | `i8` | feature-transformer FullThreats weights (big net) |
| `PSQTWeightType` | `i32` | PSQT weights (both feature sets) |
| `TransformedFeatureType` | `u8` | feature-transformer output (input to fc_0) |
| `IndexType` | `u32` | feature indices |

Layer-local types (`layers/affine_transform*.h`):

| | Rust type |
|---|---|
| affine weights (fc_0, fc_1, fc_2) | `i8` |
| affine biases | `i32` |
| affine output | `i32` |
| ReLU output | `u8` |

## Global constants

| constant | value | meaning |
|---|---|---|
| `OutputScale` | 16 | final network output is divided by this to get centipawn-ish `Value` |
| `WeightScaleBits` | 6 | affine outputs are `>> 6` inside ClippedReLU; i.e. layer weights are fixed-point with 6 fractional bits |
| `CacheLineSize` | 64 | alignment used for all weight arrays and buffers |
| `MaxSimdWidth` | 32 | padding granularity for layer dimensions |

## Quantization scheme, layer by layer

**Feature transformer.** Accumulator lanes are i16. An activation of "1.0" (fully on) corresponds to:
- big net: clipped range `[0, 255]` (u8 max) — the sum `psq_acc + threat_acc` is clamped to 0..255,
- small net: `[0, 127*2]` — weights were doubled at load (see below), so the logical range 0..127 appears as 0..254.

**Pairwise multiply.** The two halves of the accumulator are multiplied elementwise: `out[j] = clamp(acc[j]) * clamp(acc[j + L1/2]) / 512` producing u8 in 0..127ish (the /512 keeps `255*255/512 ≈ 127`). This is why FT output fits in u8. Details and the SIMD version: [[07 - Feature Transformer and Accumulator]] and [[11 - SIMD Optimizations]].

**Why ×2 at load (small net only)?** The scalar math wants `a*b/128` on 0..127 inputs. The SIMD path uses `mulhi` (keeps upper 16 bits of a 16×16 product = an implicit `>>16`). To net the right shift, inputs are pre-scaled: load-time ×2 turns the needed shift into `<<7` before `mulhi`, which avoids touching the sign bit. `(2a << 7) * (2b) >> 16 = a*b/128`. Since it's baked into the *stored* weights, a scalar Rust implementation must either replicate the ×2 (and use `/512` like Stockfish's scalar fallback does) or skip the scaling and use `/128`. Recommendation: **replicate Stockfish exactly** (×2 at load, clamp to 254, `/512`) so your outputs match bit-for-bit. The big net needs no scaling: it is trained/quantized for the 0..255 clamp and `/512` directly.

**Affine layers.** `output_i32 = bias + Σ weight_i8 · input_u8`. Note the operand asymmetry: **inputs are unsigned u8, weights signed i8** — this matches the SIMD `dpbusd` (unsigned×signed) instructions. In Rust: `i32::from(w as i8) * i32::from(x as u8)`.

**ClippedReLU.** `out = clamp(in >> 6, 0, 127)` — the `>> 6` is `WeightScaleBits`, undoing the weight fixed-point.

**SqrClippedReLU.** `out = min(127, (in*in) >> 19)` where 19 = `2*WeightScaleBits + 7`. The "+7" approximates dividing by 127 with a shift by 128 — the trainer accounts for this. Use `i64` for the square (Stockfish uses `long long`).

**Skip connection scale conversion.** fc_0's 16th output (index 15) bypasses the hidden layers and is added to fc_2's output. Its "1.0" is `127 * (1 << 6)` but the output side wants "1.0" = `600 * OutputScale`, so:

```
fwd_out   = fc0_out[15] * (600 * 16) / (127 * 64)
positional = fc2_out[0] + fwd_out
```

**PSQT path.** i32 accumulators; the net's material-like output. Final:
```
psqt = (own_psqt[bucket] - opp_psqt[bucket]) / 2                  // small net
psqt = (own-opp HalfKA psqt + own-opp threat psqt) / 2            // big net
```

**Final scale.** Both outputs are divided by `OutputScale` (16) when returned as `Value`:
```
value_psqt       = psqt / 16
value_positional = positional / 16
```

## Integer-width safety for your Rust port

- Accumulator adds/subtracts: i16 wrapping is *not* expected — trained weights guarantee no overflow; still, use `i16` with plain `+`/`-` (Stockfish does; debug-mode Rust would panic on overflow if the file were corrupt — fine).
- Affine dot products: worst case `1024 × 127 × 127 ≈ 1.6e7` fits comfortably in i32.
- SqrClippedReLU square: needs 64-bit intermediate.
- PSQT: i32 everywhere.

Division note: Stockfish uses C++ integer division (truncation toward zero) in `psqt / 2`, `/ OutputScale`, and the [[10 - Big-Small Net Switching and Final Eval|final-eval formulas]] — with negative values this differs from shifting/flooring. Rust's `/` on integers also truncates toward zero, so plain `/` matches. Don't "optimize" to `>>`.

Next: [[04 - HalfKAv2_hm Feature Set]].

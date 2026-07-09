# Rust Implementation Roadmap (nnue-rs)

> A staged plan for building a Stockfish-compatible NNUE library in Rust, ordered so every stage is independently verifiable before the next. Also collects Rust-specific design advice and the full testing strategy.

## Guiding principle

**Scalar-correct first, fast later.** Stockfish itself contains a scalar fallback for every SIMD path — your milestone 1–4 implementation *is* that fallback. Every optimization then has a ground truth to diff against.

## Milestone 1 — Parser

Deliverable: load both nets into plain structs; reject bad files.

- [[02 - LEB128 Compression|LEB128 block reader]] (magic, byte count, signed decode).
- Header + hash validation per [[01 - NNUE File Format]]. Hard-code the four expected hashes ([[14 - Constants Reference]]) or implement the hash-chain functions (tiny, worth doing).
- Read FT sections (mind the big net's raw-i8 threat block and the *shared* PSQT LEB block) and 8 layer stacks.
- **Test:** file parses to exactly EOF; array lengths match; spot-check value ranges (FT biases ≈ ±210, HalfKA weights ≈ ±900).

Suggested representation: keep weights in file-natural order in `Box<[i16]>`/`Box<[i8]>`/`Box<[i32]>` inside typed structs (`FeatureTransformer<const L1: usize>`, `LayerStack`). Generics over L1 (1024/128) mirror Stockfish's templates; or just two concrete types to keep it simple while learning.

## Milestone 2 — Feature indexing

- Board representation: you need position + occupancy + attacks. Either your own bitboard module (great Rust practice: magics or classical sliders) or the `shakmaty`/`cozy-chess` crate to start.
- Implement both `make_index` functions and the LUT construction — [[06 - make_index Deep Dive]].
- `append_active_indices` for both feature sets.
- **Test:** the worked examples in [[06 - make_index Deep Dive]]; per-attacker offset totals from [[05 - FullThreats Feature Set]]; uniqueness/range property tests; mirror-symmetry tests.

## Milestone 3 — Full (non-incremental) evaluation

- Build accumulators from scratch from active indices ([[07 - Feature Transformer and Accumulator]]).
- Scalar `transform()` (remember: small net = load-time ×2 + clamp 254 + `/512`; big = add threat acc + clamp 255 + `/512`).
- Scalar layer stack forward pass ([[08 - Layer Stacks and Forward Pass]]), including the skip connection.
- Final blend ([[10 - Big-Small Net Switching and Final Eval]]).
- **Test — the big one:** compare against real Stockfish. Build the reference SF 18 (`cd src && make -j profile-build` or just `make build`) and script it over UCI: send positions, use `eval` command output (trace shows per-bucket psqt/positional for the big net — perfect for exact comparison of raw network outputs, not just the final blended value). Match `NNUE evaluation` and per-bucket tables exactly. A few hundred random positions from self-play games ≈ complete confidence.

## Milestone 4 — Incremental updates

- `AccumulatorStack` with lazy `evaluate` (forward/backward walks) — [[09 - Incremental Updates]].
- `DirtyPiece` generation in your move-maker; `DirtyThreats` generation (`update_piece_threats` logic: outgoing, incoming, discovered rays).
- Refresh rules + Finny cache for HalfKA; full threat rebuild.
- Skip the fused double-update optimizations initially.
- **Test:** random playouts; after every move assert incremental accumulator == from-scratch accumulator (both perspectives, both nets). This catches everything — run it long.

## Milestone 5 — SIMD

Options in Rust, in increasing effort:
1. **Trust the autovectorizer** for the accumulator add/sub loops (contiguous i16 adds vectorize well) — often 70% of the win for 0% of the unsafety.
2. **`std::simd` (portable SIMD, nightly)** — clean, portable, good for accumulator + ClippedReLU paths; no dpbusd though.
3. **`core::arch` intrinsics** with runtime dispatch (`is_x86_feature_detected!`) — required for `maddubs`/`dpbusd`, `mulhi/packus` transform, `find_nnz`. Gate each kernel behind a trait or function pointer; keep the scalar kernel compiled always.

Order of impact (measure, but expect): sparse fc_0 dpbusd kernel ≫ transform() packus/mulhi ≫ accumulator tiling ≫ ClippedReLU packing. Apply the [[12 - Weight Permutation and Scrambling|load-time reorderings]] only together with the kernels that need them, and re-run the milestone-3/4 equivalence tests after each kernel (SIMD vs scalar must be bit-identical).

`unsafe` policy (matches your standards): intrinsics blocks are fine — invariants are local (alignment, length multiples asserted at construction).

## Milestone 6 — Polish

- Fused double updates ([[09 - Incremental Updates]]) if you want the last few percent.
- Benchmarks: `criterion` with (a) evals/sec on a fixed game corpus replaying moves incrementally, (b) full-refresh cost. Reference point: Stockfish does roughly 1–2M evals/sec/core with search overhead included.
- Alignment: wrap weight allocations in a 64-byte-aligned buffer type before the SIMD milestone.

## Crate layout suggestion

```
nnue-rs/
  src/
    format/       // milestone 1: leb128.rs, header.rs, reader.rs
    features/     // milestone 2: half_ka_v2_hm.rs, full_threats.rs, luts.rs
    net/          // ft.rs (transformer), layers.rs, network.rs
    accumulator/  // stack.rs, dirty.rs, finny.rs
    eval.rs       // blend + selection
    simd/         // milestone 5, behind cfg/feature flags
  tests/          // stockfish parity tests (need the nets + reference outputs)
```

Check in *reference outputs* (FEN → expected psqt/positional per net, generated once from real Stockfish) so CI doesn't need Stockfish.

## Traps I'd flag ahead of time

1. **fc_1 file padding** — 32 weight columns stored for 30 inputs. Forgetting the 2 zero columns desyncs the whole file.
2. **The shared PSQT LEB block** in the big net (threat then HalfKA psqt in ONE block).
3. **The `/2` in psqt** and the `(125,131)/128` blend — easy to drop and get "almost right" evals.
4. **Small-net ×2 scaling** — without it your small-net outputs are half-ish but not exactly (clamping differs), so it *looks* like a subtle bug.
5. **Perspective ordering** — output buffer is [stm half, opponent half]; accumulator array is indexed by color, not by stm. Mixing these up gives evals that are right for one side only.
6. **`to == SQ_NONE`** in DirtyPiece for promotions; castling's rook in `remove_sq/add_sq`.
7. **Truncating division** on negatives — don't replace `/` with shifts.
8. **FullThreats sentinel** — always filter `index < 79856`; excluded threats are generated by the diff machinery too.
9. **OrientTBL conventions differ** between the two feature sets (HalfKA normalizes king to e–h, threats to a–d). They're unrelated tables that happen to share a name.

Related: everything — start at [[00 - Stockfish NNUE Overview]].

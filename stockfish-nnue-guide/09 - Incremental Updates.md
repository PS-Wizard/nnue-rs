# Incremental Updates: the Accumulator Stack

> The "EU" in NNUE. Instead of rebuilding the accumulator each node, Stockfish records what each move changed and lazily patches accumulators when an eval is actually needed. This note covers the stack, diff records, lazy evaluation, refresh paths, Finny tables, and the fused double-update tricks.

Source: `nnue/nnue_accumulator.h/.cpp`, `types.h` (diff structs), `position.h/.cpp` (diff generation).

## The AccumulatorStack

One per search thread:

```
psq_accumulators    : [AccumulatorState<HalfKAv2_hm>; MAX_PLY+1]
threat_accumulators : [AccumulatorState<FullThreats>; MAX_PLY+1]
size                : usize   // current ply depth + 1
```

Each `AccumulatorState` holds **both** a big (1024) and small (128) accumulator plus the move's diff record. On `do_move` the stack is `push()`ed and the diff written by move-making code; on `undo_move`, `pop()` just decrements `size` (states are reused). `computed[perspective]` flags mark which accumulators are valid.

Note the structure: diffs for HalfKA (`DirtyPiece`) live in the PSQ states, diffs for threats (`DirtyThreats`) in the threat states. The big net consumes both stacks; the small net only the PSQ one.

## Diff records

### DirtyPiece (HalfKAv2_hm)

```
pc, from, to           // the moving piece (to == SQ_NONE on promotion)
remove_sq, remove_pc   // capture victim (or castling rook removal)
add_sq, add_pc         // promotion piece (or castling rook add)
```

### DirtyThreats (FullThreats)

Generated during `do_move` by `Position::update_piece_threats<PutPiece, ComputeRay>()`, which is called from `put_piece` / `remove_piece` / `move_piece` / `swap_piece`. For a piece placed on / removed from square `s` it records as add/remove:

1. **Outgoing threats**: `attacks(pc, s, occupied) & occupied` → one record per victim.
2. **Incoming threats**: all attackers of `s` (knights/pawns/kings by pseudo-attack; sliders by real attacks).
3. **Discovered threats** (`ComputeRay`, used by move/put/remove but not swap): for each slider attacking `s`, the piece *behind* `s` on the ray gains/loses the slider's attack. `move_piece` passes `noRaysContaining = from|to` to avoid double-counting the ray through both move squares.

Each record is a packed u32 `DirtyThreat` (attacker piece+square, victim piece+square, add flag — bit layout in [[05 - FullThreats Feature Set]]). Also tracked: `us`, `prevKsq`, `ksq`, and bitboards `threatenedSqs`/`threateningSqs` of squares involved in **added** threats (used by the fused-update heuristic below).

Order matters within do_move: e.g. for a capture, the victim is removed (recording its threat deletions) before the mover lands.

## Lazy evaluation flow

`FeatureTransformer::transform()` calls `accumulatorStack.evaluate(pos, ft, cache)` which, per feature set and per perspective (`evaluate_side`):

```
1. Walk back from the top: find_last_usable_accumulator
   — stop at the first state whose accumulator is computed, OR
   — at a state whose diff requires_refresh(perspective) (can't cross it), stop there.
2. If that state is computed:
       forward_update_incremental: apply diffs from it up to the top.
   Else:
       refresh the TOP state from scratch/cache, then
       backward_update_incremental: walk *down* from the top re-deriving
       older states (applying diffs in reverse), so future forward walks
       find computed ancestors.
```

Backward updating works because updates are invertible: applying a diff backward means swapping its added/removed lists.

### Refresh triggers (per feature set)

| feature set | requires_refresh | refresh method |
|---|---|---|
| HalfKAv2_hm | own king moved (`diff.pc == our king`) | Finny-table diff (`update_accumulator_refresh_cache`) |
| FullThreats | mover's own king crossed d/e file boundary (`(ksq^prevKsq) & 4`) | full rebuild from active features (`update_threats_accumulator_full`) |

### Finny tables (`AccumulatorCaches`)

Cache entry per `(king square, perspective)` (2×64 entries per net): the accumulator *and the exact board* it corresponds to. Refresh procedure:

```
entry = cache[ksq][perspective]
changed  = squares where entry.pieces != pos.pieces        (SIMD board compare)
removed  = changed & entry.pieceBB   → features to subtract
added    = changed & pos.pieces()    → features to add
patch entry accumulation/psqt by those columns; copy to the target accumulator;
store the new board into the entry.
```

Because searches revisit similar structures with the same king square, the diff is typically tiny — a full refresh costs about as much as a couple of incremental updates. Entries are initialized with `accumulation = biases`, empty board (so the first use degenerates to "add every piece").

## Applying one diff (`update_accumulator_incremental`)

For HalfKA, a move yields ≤2 added + ≤2 removed features. The update is a **fused multi-operand vector loop** (`fused_row_reduce`): one pass over the L1 lanes computing e.g. `to = from + colA - colR0 - colR1`, plus the same over the 8 PSQT lanes. Variants: Add/Sub (quiet move), Add/Sub/Sub (capture), Add/Add/Sub and Add/Add/Sub/Sub for backward updates.

For threats, added/removed lists can hold tens of entries, so the update is tiled: load accumulator tile into registers, subtract all removed columns, add all added columns, store (see [[11 - SIMD Optimizations]] for tiling). Threat weight columns are i8 widened to i16 on the fly.

## Fused double updates (two plies in one pass)

When updating across two consecutive diffs, Stockfish detects patterns where intermediate work cancels:

**PSQ (`double_inc_update`):** if move 1's `to` equals move 2's `remove_sq` (the moved piece was immediately captured), the add of that piece and its removal cancel. Both diffs are combined into one Add + 2–3 Subs applied directly from state `n-1` to state `n+1`, skipping the middle accumulator entirely.

**Threats:** if move 2 captures a piece that move 1's diff had recorded threats *for* (checked via `middle.diff.threateningSqs & bb(dp2.remove_sq)`), `append_changed_indices` is run with a `FusedUpdateData` filter: threat records added in the middle diff from/to the captured piece's square are matched against the second diff's removals and both are dropped (`dp2removedOriginBoard` / `dp2removedTargetBoard` bitboards track which). Net effect: one combined add/remove list, one tiled pass, no wasted column ops.

These are pure optimizations — results are identical to applying the two diffs sequentially. **For your Rust port: implement sequential updates first; add fusion only after everything is verified** (it's worth measurable Elo/nps in Stockfish but is the fiddliest code in the file).

## Correctness invariants worth asserting in Rust

- After `evaluate()`, the top state's accumulator equals a from-scratch rebuild (test with random game playouts — this is THE integration test).
- Backward + forward updates commute with each other and with refreshes.
- HalfKA incremental adds/removes are ≤2/≤2; threat lists ≤ 80 entries.
- A Finny refresh must yield exactly the same accumulator as a cold rebuild.

Related: [[07 - Feature Transformer and Accumulator]], [[05 - FullThreats Feature Set]], [[11 - SIMD Optimizations]].

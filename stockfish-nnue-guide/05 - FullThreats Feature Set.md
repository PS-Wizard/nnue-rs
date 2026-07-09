# FullThreats Feature Set

> The new feature set powering the **big net only** in SF 18. A feature is an *attack relationship*: "piece X on square F attacks piece Y on square T." 79,856 dimensions; up to 128 active at once.

Source: `nnue/features/full_threats.h/.cpp`. Hash: `0x8F234CB8`. Name: `"Full_Threats(Friend)"`. In the big net's feature transformer these features **add into the same accumulator** as [[04 - HalfKAv2_hm Feature Set|HalfKAv2_hm]] features (see [[07 - Feature Transformer and Accumulator]]).

## What counts as a threat

For every piece on the board, every piece it attacks (on an occupied square, using real board occupancy for sliders) *may* be a feature. "Attacks" includes friendly pieces (defenses!) — the piece colors are both encoded, so the net distinguishes attacks from defenses.

Two exclusion mechanisms cut the space:

### 1. Excluded attacker→victim type pairs (`map` table)

`map[attackerType-1][victimType-1]`, `-1` = excluded; otherwise it's a compact 0-based "valid target slot":

| attacker \ victim | P | N | B | R | Q | K | valid types |
|---|---|---|---|---|---|---|---|
| Pawn   | 0 | 1 | –1 | 2 | –1 | –1 | 3 |
| Knight | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| Bishop | 0 | 1 | 2 | 3 | –1 | 4 | 5 |
| Rook   | 0 | 1 | 2 | 3 | –1 | 4 | 5 |
| Queen  | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| King   | 0 | 1 | 2 | 3 | –1 | –1 | 4 |

(Rationale: pawn-attacks-bishop/queen/king and minor/rook/king-attacks-queen etc. are either almost always winning tactics the search sees anyway, or impossible/illegal-ish; dropping them saves space.) Victim color doubles the slot count: `numValidTargets = 2 × (valid types)` → per attacker type: P 6, N 12, B 10, R 10, Q 12, K 8.

### 2. Same-type deduplication (`semi_excluded`)

If attacker and victim are the same piece type, the threat is usually **symmetric** (knight attacks knight ⟺ that knight attacks it back), so it would be encoded twice. Rule: when `attackerType == attackedType` and (they're enemy pieces, or the type isn't pawn), the feature is only emitted when `from_oriented > to_oriented`. (Friendly pawn "attacking" a friendly pawn is *not* symmetric — a pawn defends a pawn diagonally ahead — so both directions stay.)

Implementation detail: `make_index` returns `Dimensions` (79856) as a sentinel for excluded features; callers filter with `if (index < Dimensions)`.

## Index space layout (verified by independent recomputation)

Features are grouped by (oriented) attacker piece, in the order W_PAWN, W_KNIGHT, …, W_KING, B_PAWN, …, B_KING. Within an attacker: by victim (color, mapped type), then by from-square, then by to-square rank among that piece's pseudo-attacks.

`cumulativePieceOffset` = total pseudo-attack (from,to) pairs for that piece over all from-squares — pawns count only from-squares A2..H7:

| attacker | base offset | pairs per victim-slot | victim slots |
|---|---|---|---|
| W_PAWN | 0 | 84 | 6 |
| W_KNIGHT | 504 | 336 | 12 |
| W_BISHOP | 4,536 | 560 | 10 |
| W_ROOK | 10,136 | 896 | 10 |
| W_QUEEN | 19,096 | 1,456 | 12 |
| W_KING | 36,568 | 420 | 8 |
| B_PAWN | 39,928 | 84 | 6 |
| B_KNIGHT | 40,432 | 336 | 12 |
| B_BISHOP | 44,464 | 560 | 10 |
| B_ROOK | 50,064 | 896 | 10 |
| B_QUEEN | 59,024 | 1,456 | 12 |
| B_KING | 76,496 | 420 | 8 |
| **total** | **79,856** | | |

Note "pseudo-attacks": sliders use *empty-board* rays for the index space (so the space covers every geometrically possible pair), but *active* features are generated with real occupancy.

The full index formula lives in [[06 - make_index Deep Dive]].

## Orientation

`orientation = OrientTBL[ksq] ^ (56 * perspective)`, applied by XOR to both squares. For FullThreats, `OrientTBL[ksq] = SQ_A1 (0)` when the perspective's king is on files a–d, `SQ_H1 (7)` on files e–h — i.e. the king is normalized to files **a–d** (opposite convention to HalfKAv2_hm, which normalizes to e–h; they're independent tables). Black perspective also flips ranks (`^56`) and swaps piece colors (`piece ^ 8`).

Note there is **no king-square bucket** in the threat index — the king square only affects mirroring. That's why the refresh rule is cheap:

## Refresh rule

```cpp
requires_refresh(diff, perspective) =
    perspective == diff.us            // only the side that moved
    && (diff.ksq & 4) != (diff.prevKsq & 4)   // king crossed the d/e file boundary
```

Only when *that perspective's own* king crosses between files a–d and e–h does the mirroring flip, forcing a **full rebuild** (`update_threats_accumulator_full` — there is no Finny cache for threats). All other king moves update incrementally.

## Active feature generation (`append_active_indices`)

For each color (perspective's own color first), for each piece type:
- **Pawns:** batch via bitboard shifts — `shift_NE(pawns) & occupied` and `shift_NW(pawns) & occupied` give all pawn-attack targets at once.
- **Others:** for each piece square `from`, `attacks_bb(pt, from, occupied) & occupied` gives victims; emit `make_index(persp, attacker, from, to, piece_on(to), ksq)` for each, filtered by `< Dimensions`.

`MaxActiveDimensions = 128`.

## Changed features — DirtyThreats

The diff type is a list of per-threat add/remove records generated during `do_move` by `Position::update_piece_threats` (see [[09 - Incremental Updates]] for how):

```cpp
struct DirtyThreat {          // packed into one u32
    // bits 0..7   pc_sq          (attacker square)
    // bits 8..15  threatened_sq  (victim square)
    // bits 16..19 threatened_pc  (victim piece)
    // bits 20..23 pc             (attacker piece)
    // bit  31     add            (1 = feature added, 0 = removed)
};
struct DirtyThreats {
    ValueList<DirtyThreat, 96> list;    // max 80 real + 16 slack for masked SIMD stores
    Color    us;                        // side that made the move
    Square   prevKsq, ksq;              // that side's king before/after
    Bitboard threatenedSqs, threateningSqs;  // squares involved in *added* threats
};
```

Bound: a piece has ≤ 8 outgoing and ≤ 16 incoming attacks, and moving it can reveal ≤ 8 discovered attacks → a move changes at most `(8+16)*3 + 8 = 80` features (a move touches ≤ 3 squares: from, to, capture square).

`append_changed_indices` simply converts each record via `make_index` into the added/removed lists (with an optional fused-update dedup path — see [[09 - Incremental Updates]]).

## Weight array addressing

```
threatWeights[f * L1 + j]          // i8 !  (widened to i16 when accumulating)
threatPsqtWeights[f * 8 + bucket]  // i32
```

The i8 weights make the 81.7 MB table half the size it would be at i16 and let SIMD widen on the fly.

Related: [[06 - make_index Deep Dive]], [[07 - Feature Transformer and Accumulator]], [[09 - Incremental Updates]].

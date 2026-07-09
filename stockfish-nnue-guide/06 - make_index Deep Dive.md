# make_index Deep Dive

> `make_index` is the bridge between the board and the network: it maps a board fact (a piece on a square / an attack pair) to the **feature index** used to address the weight arrays: `weights[index * L1 ..]`, `psqtWeights[index * 8 ..]`. Getting these functions bit-exact is the single most important correctness task in a reimplementation — one off-by-one and every evaluation is garbage.

Both feature sets have their own `make_index`. Both take a `perspective` — every position is encoded twice, once from white's view, once from black's, feeding the two accumulator halves ([[07 - Feature Transformer and Accumulator]]).

## Shared machinery: orientation by XOR

Square encoding: `sq = rank*8 + file`, A1=0, B1=1, … H1=7, A2=8, … H8=63.

- `sq ^ 7` mirrors **files** (a↔h, b↔g, …) — same rank.
- `sq ^ 56` mirrors **ranks** (1↔8, …) — same file, i.e. rotates the board for the black perspective.
- `piece ^ 8` flips a piece's **color** (Piece encoding: `color*8 + type`; W_PAWN=1 … W_KING=6, B_PAWN=9 … B_KING=14).

The black perspective applies `^56` to squares (and `^8` to pieces for FullThreats) so that "my pawn advancing" always looks the same to the net regardless of which side it is. The file-mirror (`^7`) is applied conditionally based on the king's file to exploit the board's left/right symmetry, halving (HalfKA) or constraining (Threats) the index space.

---

## HalfKAv2_hm::make_index

```cpp
IndexType make_index(Color perspective, Square s, Piece pc, Square ksq) {
    const IndexType flip = 56 * perspective;                 // rank-flip for black
    return (IndexType(s) ^ OrientTBL[ksq] ^ flip)            // oriented piece square (0..63)
         + PieceSquareIndex[perspective][pc]                 // piece plane base (0,64,...,640)
         + KingBuckets[int(ksq) ^ flip];                     // king bucket base (bucket*704)
}
```

Step by step:

1. **flip** = 0 for white perspective, 56 for black.
2. **OrientTBL[ksq]** = 7 if the perspective's king is on files a–d else 0. Note it's indexed by the *raw* king square; the table is laid out so this works for both perspectives combined with `flip`. Net effect: from its own point of view, the king always sits on files e–h.
3. Piece square becomes `s ^ OrientTBL[ksq] ^ flip` — mirrored and/or rotated.
4. **PieceSquareIndex[perspective][pc]** — the 704-wide plane table from [[04 - HalfKAv2_hm Feature Set]] ("our pawn"=0, "their pawn"=64, …, king=640). Indexing by perspective handles the us/them swap; no `pc ^ 8` needed here.
5. **KingBuckets[ksq ^ flip]** — bucket index (0..31) × 704, from the king's own-perspective square. Mirrored files share buckets, which is what makes `Dimensions = 32 × 704 = 22,528`.

### Worked example 1

White pawn on **e4**, white king on **g1**, white perspective:
- `flip = 0`; king file g ⇒ `OrientTBL[g1] = 0` (no mirror).
- oriented square = e4 = 28.
- plane = our pawn = 0.
- `KingBuckets[g1=6]` = bucket 29 ⇒ base `29 × 704 = 20416`.
- **index = 28 + 0 + 20416 = 20444.**

### Worked example 2 (black perspective of the same pawn)

Same white pawn on e4, black king on **e8**, black perspective:
- `flip = 56`; `OrientTBL[e8]`: e8 is file e ⇒ 0 (no mirror).
- oriented square = `28 ^ 0 ^ 56 = 36` (e5 — the pawn seen from black's rotated board).
- plane = `PieceSquareIndex[BLACK][W_PAWN]` = *their* pawn = 64.
- king bucket: `KingBuckets[e8 ^ 56 = e1 = 4]` = bucket 31 ⇒ `31 × 704 = 21824`.
- **index = 36 + 64 + 21824 = 21924.**

### Worked example 3 (mirroring kicks in)

White knight on **b1**, white king on **c1**, white perspective:
- king file c (a–d) ⇒ `OrientTBL[c1] = 7`: mirror files.
- oriented square = `b1=1 ^ 7 = 6` = g1.
- plane = our knight = 128.
- king bucket: `KingBuckets[c1=2]` = bucket 30 ⇒ 21120. (b1 file mirror maps king to f1 conceptually; the *table itself* already encodes the mirror, both c1 and f1 give bucket 30.)
- **index = 6 + 128 + 21120 = 21254.**

---

## FullThreats::make_index

```cpp
IndexType make_index(Color perspective, Piece attacker, Square from,
                     Square to, Piece attacked, Square ksq) {
    const int8_t orientation   = OrientTBL[ksq] ^ (56 * perspective);
    unsigned     from_oriented = uint8_t(from) ^ orientation;
    unsigned     to_oriented   = uint8_t(to)   ^ orientation;

    int8_t   swap              = 8 * perspective;
    unsigned attacker_oriented = attacker ^ swap;   // flip piece colors for black
    unsigned attacked_oriented = attacked ^ swap;

    return index_lut1[attacker_oriented][attacked_oriented][from_oriented < to_oriented]
         + offsets[attacker_oriented][from_oriented]
         + index_lut2[attacker_oriented][from_oriented][to_oriented];
}
```

Three precomputed tables sum to the index. Here `OrientTBL[ksq]` = 0 for king files a–d, 7 for e–h (king normalized to a–d — opposite of HalfKA, deliberately an independent convention).

### Table 1 — `index_lut1[attacker][attacked][from < to]`: the (attacker, victim) block base

```
feature_base = base_offset(attacker)                              // table in [[05 - FullThreats Feature Set]]
             + (color(attacked) * numValidTargets(attacker)/2 + map[atkType][vicType])
               * pairs_per_slot(attacker)
```

- If `map` is `-1` (excluded pair): both entries = `Dimensions` (79856) → caller drops the feature.
- The `[from < to]` axis implements same-type dedup: for a symmetric pair (same type; enemy, or non-pawn) the entry for `from < to` is `Dimensions`, so only the `from > to` direction survives.
- `color(attacked)` here is the *oriented* victim color: 0 = perspective's own piece (a defense!), 1 = enemy piece.

### Table 2 — `offsets[attacker][from]`: from-square base within the block

Cumulative count of pseudo-attack targets over all previous from-squares for this piece type. E.g. for a knight: `offsets[N][s] = Σ_{q<s} popcount(knightAttacks(q))`. Pawns: only from-squares A2–H7 contribute (a pawn can't stand on rank 1/8); white and black pawns have different attack directions and thus different tables.

### Table 3 — `index_lut2[attacker][from][to]`: to-square rank

The rank of `to` among the pseudo-attack targets of `attacker` on `from`, in square order:
```
index_lut2[p][from][to] = popcount(pseudoAttacks(p, from) & ((1 << to) - 1))
```
Sliders use **empty-board** rays here (the index space ignores occupancy; occupancy only decides which features are *active*).

### Worked example (fully verified numerically)

White knight on **f3** attacks black pawn on **e5**; white king on **g1**; white perspective.

1. King on file g ⇒ `OrientTBL[g1] = 7`; perspective white ⇒ `orientation = 7` (mirror files).
2. `from = f3 = 21 → 21^7 = 18 = c3`; `to = e5 = 36 → 36^7 = 35 = d5`.
3. `swap = 0`: attacker stays W_KNIGHT (2), victim stays B_PAWN (9).
4. `index_lut1[W_KNIGHT][B_PAWN][18 < 35 → 1]`:
   - not excluded (`map[N][P] = 0`), not same type ⇒ both directions valid.
   - `= 504 + (1 × 6 + 0) × 336 = 504 + 2016 = 2520` (enemy-pawn victim slot of the knight block).
5. `offsets[W_KNIGHT][c3=18]` = knight-attack counts of squares A1..B3 summed = **74**.
6. knight attacks from c3 (square order): b1, d1, a2, e2, a4, e4, b5, **d5** → d5 is the 8th ⇒ `index_lut2 = 7`.
7. **index = 2520 + 74 + 7 = 2601.**

### Dedup example

Two enemy knights on c3 and d5 attack each other. Same type, enemy ⇒ `semi_excluded`. Only the direction with `from_oriented > to_oriented` (the d5 knight → c3 knight record) produces a valid index; the other returns 79856 and is skipped. Without this, the accumulator would double-count one logical relationship.

---

## Implementation notes for Rust

- Build all LUTs at startup (or `const fn` / `build.rs`): `index_lut1` is `[16][16][2] u32` (index by raw Piece values 0..15, entries for invalid pieces unused), `offsets` is `[16][64] u32`, `index_lut2` is `[16][64][64] u8`. Total < 70 KB — cache-friendly.
- Replicate the sentinel pattern (`>= DIMENSIONS` means skip). It lets the hot loop stay branch-light: compute unconditionally, filter once.
- Unit-test against known values: the worked examples above, plus property tests — for every (attacker,victim,from,to) with `to ∈ pseudoAttacks(attacker,from)` and map ≥ 0, indices must be unique and `< 79856`; summed counts per attacker must match the table in [[05 - FullThreats Feature Set]].
- For HalfKA: iterate all (ksq, pc, sq) triples; assert index `< 22528`; assert mirror-symmetry (mirroring the whole position along the vertical axis yields identical indices).

Related: [[04 - HalfKAv2_hm Feature Set]], [[05 - FullThreats Feature Set]], [[07 - Feature Transformer and Accumulator]].

# HalfKAv2_hm Feature Set

> "Half **K**ing-**A**ll pieces, version 2, **h**orizontally **m**irrored." The piece-square feature set used by **both** networks. 22,528 dimensions; ≤ 32 active at once (one per piece on the board).

Source: `nnue/features/half_ka_v2_hm.h/.cpp`. Hash: `0x7F234CB8`. Name string: `"HalfKAv2_hm(Friend)"`.

## Concept

A feature is: *"from perspective P, with P's king in king-bucket B, there is a piece of type/color T on square S."* Every piece on the board (including both kings) activates exactly one feature per perspective. Two perspectives (white, black) are maintained as two independent accumulator halves — "Half" refers to this.

## The three ingredients

`index = oriented_square + piece_plane + king_bucket_base` — computed by `make_index` (full walkthrough with worked examples in [[06 - make_index Deep Dive]]).

### 1. Piece planes (`PieceSquareIndex`)

11 planes of 64 squares (`PS_NB = 704`). From a given perspective, "W" means *our* color:

| plane | contents | base |
|---|---|---|
| 0 | our pawn | 0 |
| 1 | their pawn | 64 |
| 2 | our knight | 128 |
| 3 | their knight | 192 |
| 4 | our bishop | 256 |
| 5 | their bishop | 320 |
| 6 | our rook | 384 |
| 7 | their rook | 448 |
| 8 | our queen | 512 |
| 9 | their queen | 576 |
| 10 | **any king** (shared!) | 640 |

Both kings share one plane — that's the "v2" economization (the opponent king's position is still encoded since it's that perspective's own-king bucket for the *other* half).

### 2. King buckets (`KingBuckets`)

The own-king square (after orientation) selects one of **32 buckets**; the bucket base is `bucket * 704`. The table is mirrored so files a..d and h..e map identically (the "hm" = horizontally mirrored part), which halves the dimension count:

```
Dimensions = 64 * 704 / 2 = 22,528
```

Bucket layout (viewing from white, before mirroring; value = bucket index, base = value × 704):

```
rank 1:  28 29 30 31 | 31 30 29 28
rank 2:  24 25 26 27 | 27 26 25 24
rank 3:  20 21 22 23 | 23 22 21 20
rank 4:  16 17 18 19 | 19 18 17 16
rank 5:  12 13 14 15 | 15 14 13 12
rank 6:   8  9 10 11 | 11 10  9  8
rank 7:   4  5  6  7 |  7  6  5  4
rank 8:   0  1  2  3 |  3  2  1  0
         a  b  c  d    e  f  g  h
```

(The table is indexed by `ksq ^ (56*perspective)` — the king square from the perspective's own point of view.)

### 3. Orientation (`OrientTBL`)

If the (own) king is on files **a–d**, the whole board is mirrored horizontally (`square ^= 7`) so that the king effectively always sits on files e–h. The table stores the XOR mask per king square: `SQ_H1` (7) for king files a–d, `SQ_A1` (0) for e–h. Additionally the black perspective flips ranks: `square ^= 56`.

## Active features

```
for each square s with a piece pc:
    active.push(make_index(perspective, s, pc, ksq_of_perspective))
```
Max 32 active (`MaxActiveDimensions = 32`).

## Changed features (incremental updates)

The diff type is `DirtyPiece`:

```cpp
struct DirtyPiece {
    Piece  pc;              // the moving piece (never NO_PIECE)
    Square from, to;        // to == SQ_NONE for promotions (pawn vanishes)
    Square remove_sq, add_sq;  // captured piece / promoted piece / castling rook
    Piece  remove_pc, add_pc;
};
```

`append_changed_indices` maps it to removed/added feature indices:
- remove `(pc, from)`; add `(pc, to)` if `to != SQ_NONE`
- if `remove_sq != SQ_NONE`: remove `(remove_pc, remove_sq)` (captures; promotions reuse this? no — promotions: `to = SQ_NONE`, `add_sq/add_pc` = promoted piece, captures use `remove_*`)
- if `add_sq != SQ_NONE`: add `(add_pc, add_sq)`
- castling encodes the rook via `remove_sq/add_sq` and the king via `from/to`

So a normal move = 1 removed + 1 added; capture = 2 removed + 1 added; promotion-capture = 2 removed + 1 added (pawn removed, victim removed, new piece added); castling = 2 removed + 2 added.

## Refresh rule

```cpp
requires_refresh(diff, perspective) = (diff.pc == make_piece(perspective, KING))
```

**Any own-king move invalidates that perspective's accumulator** (the king bucket and possibly orientation change, which re-indexes *every* feature). Refreshes are served from the Finny-table cache — see [[09 - Incremental Updates]].

## Weight array addressing

For feature index `f`, perspective-half accumulator lane `j` (see [[07 - Feature Transformer and Accumulator]]):
```
weights[f * L1 + j]        // i16
psqtWeights[f * 8 + bucket]  // i32
```

Related: [[05 - FullThreats Feature Set]], [[06 - make_index Deep Dive]].

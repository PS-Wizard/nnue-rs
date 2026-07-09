# Big/Small Net Switching and Final Evaluation

> How Stockfish decides which network to run, how the two network outputs (psqt, positional) are blended, and every adjustment between raw network output and the final `Value`. All constants below are from SF 18's `evaluate.cpp` (these get retuned every release).

## Step 0 — simple_eval

A one-line material count from the side to move's view:

```
simple_eval(pos) = 208 * (ourPawns - theirPawns) + (ourNonPawnMaterial - theirNonPawnMaterial)
```

Piece values (`types.h`): Pawn 208, Knight 781, Bishop 825, Rook 1276, Queen 2538 (non-pawn material = sum over N/B/R/Q).

## Step 1 — network selection

```
use_smallnet = |simple_eval(pos)| > 962
```

Lopsided material ⇒ the small net is accurate enough and much faster. Balanced ⇒ big net.

```
(psqt, positional) = (smallNet ? networks.small : networks.big).evaluate(pos)
nnue = (125 * psqt + 131 * positional) / 128
```

Note the asymmetric blend — the positional half is trusted slightly more than the material half.

### Fallback re-evaluation

```
if smallNet && |nnue| < 277:
    (psqt, positional) = networks.big.evaluate(pos)
    nnue = (125 * psqt + 131 * positional) / 128
    smallNet = false
```

If the small net says "actually this is close", the position deserves the big net after all. (This is why both accumulators are maintained on the stack — either net may be consulted at any node.)

## Step 2 — complexity adjustment

The disagreement between the two output heads measures how "tricky" the position is:

```
nnueComplexity = |psqt - positional|
optimism += optimism * nnueComplexity / 476     // trust optimism more in complex positions
nnue     -= nnue     * nnueComplexity / 18236   // trust the eval slightly less
```

(`optimism` is a search-supplied bias toward the side to move; pass 0 for a bare evaluator.)

## Step 3 — material scaling + optimism blend

```
material = 534 * pos.count<PAWN>() + pos.non_pawn_material()      // BOTH sides summed
v = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871
```

Evals scale up when there's more material on the board (a +1 with queens on is worth more than +1 in a pawn ending).

## Step 4 — 50-move-rule damping and clamp

```
v -= v * pos.rule50_count() / 199
v = clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)
```

Shuffling drags the eval toward 0 linearly with the halfmove clock. The clamp keeps static evals out of the tablebase/mate score ranges (`VALUE_TB_WIN_IN_MAX_PLY = 32000+1-2*246`... in practice: keep |v| below ~31k; exact enum values in `types.h`).

Also note the precondition: `evaluate()` asserts **not in check** (search never calls static eval in check).

## How FullThreats "interacts"

FullThreats is not a separate network — it's extra input features for the big net's feature transformer. Its contributions enter through:
1. the accumulator sum inside `transform()` (`clamp(psqAcc + threatAcc, 0, 255)`), and
2. the PSQT average: `psqt = (halfKA_psqt_diff + threat_psqt_diff) / 2`.

Everything downstream (layer stacks, blending) is unchanged. See [[07 - Feature Transformer and Accumulator]].

## Minimal evaluator pseudocode (what your Rust crate exposes)

```rust
fn evaluate(pos: &Position, nets: &Networks, stack: &mut AccStack, caches: &mut Caches,
            optimism: i32) -> i32 {
    let small = simple_eval(pos).abs() > 962;
    let (mut psqt, mut positional) =
        if small { nets.small.evaluate(pos, stack, &mut caches.small) }
        else     { nets.big.evaluate(pos, stack, &mut caches.big) };
    let mut nnue = (125 * psqt + 131 * positional) / 128;
    if small && nnue.abs() < 277 {
        (psqt, positional) = nets.big.evaluate(pos, stack, &mut caches.big);
        nnue = (125 * psqt + 131 * positional) / 128;
    }
    let complexity = (psqt - positional).abs();
    let optimism = optimism + optimism * complexity / 476;
    let nnue = nnue - nnue * complexity / 18236;
    let material = 534 * pos.pawn_count() + pos.non_pawn_material_total();
    let mut v = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871;
    v -= v * pos.rule50() as i32 / 199;
    v.clamp(TB_LOSS_BOUND + 1, TB_WIN_BOUND - 1)
}
```

All divisions truncate toward zero (Rust `/` matches C++). Signs matter: all values are from the **side to move's** perspective.

Related: [[08 - Layer Stacks and Forward Pass]], [[14 - Constants Reference]].

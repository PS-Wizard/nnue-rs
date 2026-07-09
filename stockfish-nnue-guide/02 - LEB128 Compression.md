# LEB128 Compression in NNUE files

> Several sections of the [[01 - NNUE File Format|.nnue file]] store integers as **signed LEB128** ("Little Endian Base 128") — a variable-length encoding where small-magnitude values take 1 byte. Since most NNUE weights are small, this cuts the file size dramatically (e.g. the small net's 2.88M i16 weights compress to ~3 MB instead of 5.8 MB).

## Block format

A LEB-compressed section ("block") in the file looks like:

```
17 bytes  magic string  "COMPRESSED_LEB128"    (no NUL terminator)
 4 bytes  u32 byte_count      — size of the compressed payload that follows
 N bytes  compressed payload  (N = byte_count)
```

The payload decodes to a known, fixed number of integers (the array size is known from the architecture — the format is not self-describing). Decode exactly `count` values; when you're done you must have consumed exactly `byte_count` bytes (Stockfish asserts `bytes_left == 0`).

**Shared blocks:** one block can hold multiple arrays back-to-back. The big net's PSQT section is a single block containing `threatPsqtWeights` (638,848 × i32) followed by `psqtWeights` (180,224 × i32). Decode continues seamlessly across the array boundary.

## Signed LEB128 encoding

Each value is written 7 bits at a time, low bits first. Bit 7 (0x80) of each byte = "more bytes follow". On the final byte, bit 6 (0x40) is the sign bit for sign extension.

### Decoding one value (what you implement)

```rust
fn read_signed_leb128(bytes: &mut impl Iterator<Item = u8>) -> i32 {
    let mut result: u32 = 0;
    let mut shift = 0;
    loop {
        let byte = bytes.next().unwrap();
        result |= ((byte & 0x7f) as u32) << (shift % 32);
        shift += 7;
        if byte & 0x80 == 0 {
            // sign-extend if the sign bit of the last byte is set
            // and we haven't already filled 32 bits
            if shift < 32 && (byte & 0x40) != 0 {
                result |= !((1u32 << shift) - 1);
            }
            return result as i32;
        }
    }
}
```

Notes matching Stockfish's decoder exactly (`nnue_common.h:read_leb_128_detail`):
- Stockfish only implements this for signed types ≤ 32 bits (i8, i16, i32 targets).
- The `shift % 32` and `shift >= 32` guards mirror Stockfish's overflow handling; in practice well-formed values for an iN target never exceed ceil(N/7)+1 bytes.
- After decoding into a wider intermediate, the value is stored into the target type (i16 for FT weights/biases, i32 for PSQT). Values always fit; no saturation is done.

### Worked examples

| bytes | decoding | value |
|---|---|---|
| `0x05` | no cont bit, bit6 clear | `5` |
| `0x7B` | no cont, bit6 **set** → sign-extend: `0x7B` = 123, extend from bit 7 | `-5` |
| `0xC7 0x01` | `0x47`(low7) + `0x01`<<7 = 199... : first byte cont set, low7=0x47; second byte 0x01, bit6 clear → result = 0x47 \| (0x01<<7) = 199 | `199` |
| `0xB9 0x7E` | low7=0x39; second 0x7E, bit6 set → 0x39 \| (0x7E<<7) = 0x3F39, sign-extend from bit 14 | `-199` |

### Encoding (only needed if you want to re-export nets)

```
loop {
    byte = value & 0x7f
    value >>= 7          // arithmetic shift!
    done = (byte & 0x40 == 0) ? (value == 0) : (value == -1)
    write(byte | (done ? 0 : 0x80))
    if done { break }
}
```

## Buffered decoding (performance)

Stockfish reads the payload through an 8 KiB buffer rather than byte-at-a-time stream reads. In Rust the natural equivalent is to read the whole `byte_count` payload into a `Vec<u8>` (or memory-map the file) and decode from the slice — decoding 23M values is then a tight scalar loop, well under a second. Don't bother with SIMD here; loading happens once.

## Which sections use LEB128?

| net | section | LEB? |
|---|---|---|
| both | FT biases | ✅ |
| big | FT threatWeights (i8) | ❌ raw bytes |
| both | FT weights (i16) | ✅ |
| big | threatPsqt + psqt (single block) | ✅ |
| small | psqtWeights | ✅ |
| both | all layer-stack weights/biases | ❌ raw little-endian |

See [[01 - NNUE File Format]] for the full map, [[03 - Data Types and Quantization]] for what the decoded integers mean.

#!/usr/bin/env python3
"""TP1 decode tiles for the engine's block-32 FP8 TileLang linears.

The base engine's DECODE_FP8_CONFIGS are keyed on TP4 shapes, so at TP1 every
non-MoE linear falls back to the signature default (block_K=128, num_stages=3),
and WO-A's 16-row bucket runs 64x64x64 one stage deep. Measured on one GB10 at
the TP1 decode shapes (L2-cold, 7 rows):

  wo_b 5120x8192, wq_b 32768x1280,
  shared gate_up 4608x5120, shared down 5120x2304   block_K=256, ns=3: -3.8 .. -4.9 %
  wo_a 8 groups, 16-row bucket                       32x256, ns=2:      -14.3 %

In the served model: 144.8 -> 141.4 ms per decode step (two boots per arm).
Every tile is bitwise identical to the one it replaces: K is still summed in
ascending order. Only decode buckets change; the prefill tiles are untouched.

usage: tp1_decode_tiles.py <.../vllm/models/deepseek_v4_1/tilelang>
"""
import ast
import sys
from pathlib import Path

d = Path(sys.argv[1])

gemm = d / "gemm.py"
s = gemm.read_text()
anchor = """    # TP4 DSpark main projection.
    (6400, 6144): {
        16: dict(block_N=32, block_K=256, num_stages=4, threads=64),
        32: dict(block_N=32, block_K=256, num_stages=4, threads=64),
        64: dict(block_N=64, block_K=256, num_stages=2),
    },
}"""
if "(5120, 8192): {16:" not in s:
    if s.count(anchor) != 1:
        sys.exit(f"DECODE_FP8_CONFIGS anchor not found in {gemm}")
    s = s.replace(anchor, anchor[:-1] + """    # TP1 decode: the TP4 keys above never match a TP1 shape. block_K=256 halves the
    # K-block count; bitwise identical to the 128 tile (K summed in ascending order).
    (5120, 8192): {16: dict(block_N=64, block_K=256, num_stages=3)},
    (32768, 1280): {16: dict(block_N=64, block_K=256, num_stages=3)},
    (4608, 5120): {16: dict(block_N=64, block_K=256, num_stages=3)},
    (5120, 2304): {16: dict(block_N=64, block_K=256, num_stages=3)},
}""")
    gemm.write_text(s)

wo = d / "wo.py"
s = wo.read_text()
old = "_WO_A_COMMON = dict(block_N=64, num_stages=2)"
if "_WO_A_TILE_CFG" not in s:
    if s.count(old) != 1:
        sys.exit(f"_WO_A_COMMON anchor not found in {wo}")
    s = s.replace(old, old + """
# TP1 decode: WO-A's 16-row bucket at 32x256, 2 stages (-14.3 % at 7 rows, bitwise
# identical). Only the 16-row bucket moves; the prefill tile keeps _WO_A_COMMON.
_WO_A_TILE_CFG = {16: dict(block_N=32, block_K=256, num_stages=2)}""")
    n = s.count("**_WO_A_COMMON,")
    if n != 2:
        sys.exit(f"expected 2 _WO_A_COMMON call sites in {wo}, found {n}")
    s = s.replace("**_WO_A_COMMON,", "**_WO_A_TILE_CFG.get(bucket, _WO_A_COMMON),")
    wo.write_text(s)

for f in (gemm, wo):
    ast.parse(f.read_text())
print(f"TP1 decode tiles applied in {d}")

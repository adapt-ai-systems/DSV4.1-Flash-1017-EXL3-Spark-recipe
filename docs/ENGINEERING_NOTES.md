# Engineering notes

What was tried on this deployment, in the order it was tried, including the
things that lost. Every entry is a measurement on one DGX Spark (GB10, 121.7 GiB
unified, ATS), TP1, `--max-model-len 32768`, `--max-num-seqs 1`, bench7 with 512 generated
tokens over three prompts per class. Numbers are `ms/step` (kernel speed) and
`tok/s` (client-observed), because acceptance moves between runs and `tok/s`
alone is noisy.

The short version: **the pack does not fit resident, so most of the work was
deciding what to keep in memory and how to read the rest.** Everything else is
second order.

## 1. Getting it to load at all

- **The checkpoint declares `quant_method: exl3` but the engine's V4.1 model
  class gates on its native block-32 FP8 name.** The shim (`shim/`) keeps one
  config object and gives it the accepted name, returns `Exl3MoEMethod` for
  routed experts (main stack and `mtp.*`), and delegates every dense/attention
  weight to the declared block-32 FP8 format. No vLLM source file is modified.
- **The DSpark drafter re-derives its quant config from the checkpoint**, which
  loses `non_routed_quantization`; every non-routed linear then silently falls
  back to unquantized and V4.1's attention rejects it. Restating the
  declaration in both `__init__` and `from_config` fixes it. This is why the
  `--hf-overrides` payload exists at all.
- **The pack ships an EXL3 trellis output head** (`head.trellis`/`suh`/`svh`/
  `mul1`), not a BF16 `head.weight`. The shim rebuilds `lm_head` against the
  quant config and maps the four tensors onto it.
- **io_uring init fails under the default seccomp profile.** `seccomp=unconfined`
  plus `label=disable` and `memlock=-1` are requirements, not tuning.
- **Aligned prefill chunks need `--long-prefill-token-threshold 2047`.**
- **`max_position_embeddings` is 1M**, so the two shared RoPE caches are 256 MiB
  each in fp32. Trimming them to `max_model_len` frees 0.484 GiB at boot
  (`ROPE TRIM ... rows` in the log is the receipt; 0.250 GiB at 524,288).
- **`VLLM_EXL3_TRIM_AFTER_LAYER=1` is inert at this pin.** The per-layer
  `malloc_trim`/`empty_cache` hook lives in `patch_trim.py`, which patches
  `vllm_exl3/exl3.py` *inside the image*; the measured server bind-mounted
  `plugin-moe-p9b` over that directory, so the hook was shadowed and never ran.
  `serve.sh` still passes the variable (it is harmless), but do not attribute any
  memory number to it. Bake `patch_trim.py` or drop the variable.

## 2. Memory: what stays resident

Resident is 110.5 GiB of 121.7. Everything below was measured, not estimated.

| Attempt | Result |
|---|---|
| Everything resident | does not fit; un-aliasing layers 38/39 costs 4.6 GiB of transient load pressure and the guard kills the boot |
| Whole-layer aliasing (layers 38/39 + all MTP experts) | ATS-slow: about 10 % of every aliased layer's experts are touched each step and the GPU reads them through ATS at ~6.6 GB/s; ~130 ms of a ~292 ms step |
| Cold-expert alias side file (shipped) | 1,380 least-routed experts + embedding, 8.59 GiB, 64-byte aligned; **24.2 tok/s** from 13.5 |
| MTP experts resident (with 38/39 aliased) | **regression, 9.5 tok/s** (411-847 ms/step): the extra 2.8 GiB leaves 1.2 GiB available, the page cache collapses and the still-aliased layers read even slower |
| `--max-num-batched-tokens 8192` | does not fit |

The cold set is a routing-frequency selection, not a calibration. Experts the
router rarely picks are the ones whose pages can come from NVMe without hurting
the step; the membership is `configs/cold-alias-selection.json`.

## 3. Decode: the kernel

- **Grouped MoE → unified mixed-K: +9 % mean tok/s**, prose step -16 %, code
  step -29 %. One launch per layer covers every K group instead of one dispatch
  per K group, which also removes most of the routing glue launches.
- **Tile-K 16 (`MOE_TILESIZE_K`) is the single biggest kernel win: 190 → 142
  ms/step, -25 %.** It is a compile-time `#define`. At the pinned fork commit the
  header already reads 16; the image build rewrites it to `MOE_TILESIZE_K` anyway and
  fails if the result is not that value, so the build argument cannot be ignored.
- **`__launch_bounds__(256, 2)` cannot give 2 CTAs/SM on this part.** GB10
  exposes 100 KiB of shared memory per SM, and the launcher always requests
  90 KiB as dynamic smem, whatever instance it picked. Two CTAs need 186 KiB.
  Forcing it bought 128 registers and 2.3x more spilling for no extra
  concurrency, and cost 49 %. A real second CTA needs a per-instance smem launch
  argument, i.e. a code change.
- **Two CTAs/SM done properly is still slower: +12 %.** With the launcher asking
  for the instance's real dynamic smem (29,184 B instead of 92,160) plus
  `__launch_bounds__(256, 2)` and a 16-CTA grid, two CTAs do fit: 128 registers,
  176 B spill. L2-cold at 7 rows the layer went 2.646 -> 2.967 ms. The kernel is
  not short of warps.
- **Removing one of the four `bar.sync` per K-slice: -3.9 % cold, about -3.8
  ms/step in model** (2.646 -> 2.543 ms/layer; 145.4/141.3 -> 140.0/139.1 ms/step
  over two alternating pairs), bit-exact. Not shipped yet: its teacher-forced
  gate was one better and one worse than the baseline's across the two pairs.
- **The kernel already keeps enough bytes in flight.** About 188 KiB is in flight
  across the chip, against ~186 KiB needed to saturate DRAM at ~700 ns latency
  (Little's law). The same model predicts the measured speed at 3 smem stages,
  which is why stages beyond 4 do nothing. What limits it is a fixed dependent
  chain per K-slice (~367 ns whatever the bit width or stage count): issue
  floor ~35 %, barriers ~16 %, the rest exposed latency.
- **The trellis decode is at its instruction floor**: 4.5 instructions per
  weight (multiply, byte sum, window extract, pack, affine), about 17 % of the
  slice loop. A bit-exact rewrite of the extraction changed 0.02 % of the SASS
  and measured the same in model (144.65 vs 144.80 ms/step).
- **`MOE_SH_STAGES 3 → 4`: -11 % on an L2-flushed microbench, +4 % in the
  model.** The cold bench rewards prefetch depth; the real step's L2 hit rate is
  already 77 %, so the extra stages buy nothing and cost issue slots. Not
  shipped. Microbench wins on this kernel need an in-model confirm.
- **`MOE_SMS_PER_EXPERT 8 → 4`: +84 %.** More barrier rounds per expert for
  fewer bytes each. 16 is a wash. 8 is the right width.
- **The kernel spills** (25.5 M local requests) at 255 registers/thread in the
  MAXBITS=8 instance the model actually runs. An earlier "no spills" reading was
  `cuobjdump` on a different instance. Cutting the frame to 240 B is 3.7 %
  faster cold; it was not tried in model.
- **An always-inline-lambda variant was a no-op under nvcc.** The build reported
  success and produced byte-identical step times; the compiler had already
  inlined the calls, so the source edit changed nothing. Check the SASS, not the
  diff.

## 4. Decode: everything around the kernel

- **CUDA graphs: +11-14 % at 512 tokens**, within noise at 128.
  `FULL_DECODE_ONLY` at `--max-num-seqs 1`.
- **`L2_PREFETCH=0` is kept**: the prefetch measured 8.7 ms/step of work that
  hid nothing (the step runs effectively single-stream, kernel overlap 1.04).
- **TP1 decode tiles for the non-MoE FP8 linears: -3.4 ms/step, shipped.** All six
  linears are `long_scoreboard`-bound at 9-36 % occupancy; the base image's tile tables
  are keyed on TP4 shapes, so TP1 ran the default `block_K=128`. A 62-config L2-cold
  sweep found `block_K=256` (-3.8 to -4.9 %) and, for WO-A's 16-row bucket, 32x256x2
  (-14.3 %); every winner is bitwise identical. In model: 141.4 vs 144.8 ms/step over
  two alternating boots per arm.
- **An earlier non-MoE overlay (split-K workspace + a WO-A tile): dead even, twice** (-3.3 % at a 30 % non-MoE share,
  +0.6 % at 42 %). Not shipped: it does not pay for its mount.
- **A tilelang split-K variant failed its correctness gate**: 78 % top-1
  agreement and max |dlogprob| 1.86 against the non-speculative reference, i.e.
  19 points of agreement lost for a 0.65 % step-time win. Blocked pending a fix.
- **Adaptive verification depth policy: -3.5 % tok/s, +7 % ms/step**, and the
  engine never logged a depth histogram, so the policy was inert. Not shipped.
- **The DSpark drafter is a small share of the step: 9.2 of 139.4 ms.** It runs
  one pass of the three MTP layers over 5 query rows (5.4 ms), one draft head
  launch (2.8 ms) and five sequential NVFP4 Markov steps (0.5 ms in total). The
  MTP experts are already resident (none are in the cold alias) and the draft
  graph is fully captured. Even a free drafter would only take 33.5 -> 35.9
  tok/s; the step is the 40 main layers verifying 6 rows (124.4 ms).
- **k sweep at the shipped step time: k=5 wins.** k=6 costs 1.5 ms more per step
  for +0.04 tok/step; k=7 costs +17 ms for +0.55. Re-sweep if the step ever
  drops under ~120 ms.

## 5. Where it stands

About 140-147 ms/step (140.5-142.0 at `--max-model-len 32768`, 147.2 at the
default 524288), 4.1-4.8 accepted tokens/step, 27-35 tok/s. Per-step attribution
by CUDA graph (139.4 ms): 40 main layers verifying 6 rows 124.4 ms (89 %), verify
head and sampling 2.9 ms, drafter 9.2 ms, glue 0.5 ms. Inside the main layers the
MoE is the dominant cost; the shared expert runs on a second stream and is
mostly hidden, so earlier anatomies that added it to the critical path
over-counted it.

Reaching 50 tok/s needs about 95 ms/step at ~4.7 tokens/step, i.e. about 45 ms
out of the step. What is left, by evidence:

1. **The MoE kernel's per-slice dependent chain.** Not bandwidth in flight, not
   warps, not decode instructions (all measured above). The open variants
   shorten the chain: fewer CTA-wide barriers through a column-private
   reduction and per-warp ownership of the weight stages, and a run-ahead L2
   prefetch of the next slices.
2. **The drafter** (up to -2.5 ms): a draft head restricted to part of the
   vocabulary. Emitted text is unchanged under greedy verify; acceptance may
   move.
3. **The non-MoE FP8 linears are near the memory roof** (wo_b moves 42 MB in
   0.207 ms, ~200 GB/s, after the TP1 tiles). Only fewer bytes would move them.

`dram__bytes_in_flight` does not exist on GB10 (unified LPDDR5X), so
bytes-in-flight has to be inferred from L2 throughput times latency rather than
read.

## 6. Measurement notes

- **Greedy is not bit-deterministic.** Batched atomics reorder the MoE reduce.
  Two runs of the same prompt can differ in the tail; compare logprobs, not
  strings.
- **Check what the container actually mounted after every boot.** A launcher
  that set the fork build path without exporting it made the container run the
  image's stock MoE kernel (183-198 ms/step instead of 140) with nothing in the
  logs to say so. `docker inspect` the mounts and grep the `ROPE TRIM ... rows`
  line before quoting a number.
- **Warm up and keep the disk quiet before timing.** A boot right after ~10 GB of
  build output on a 97 %-full disk ran 509.5 ms/step for its whole life; the same
  configuration after `sync`, `drop_caches` and one warmup request ran 145.4.
  Cold experts are read from NVMe on every step.
- **`ms/step` is the stable axis.** Acceptance on this box swings 3.4-5.5
  between runs on identical code, so `tok/s` across arms is not a clean
  comparison unless the tok/step column moves with it.
- **Status/naming:** every speed figure in this repo is single-stream, 512
  tokens, `ignore_eos`, three prompts per class. Prefill figures are
  repeated-text unless labelled otherwise.
# DSV4.1-Flash-1017-EXL3-Spark-recipe

Serving recipe for **vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw** on a single
NVIDIA DGX Spark (GB10, 128 GB unified memory), using vLLM. Every number below
was measured on that exact deployment; the flag set in `scripts/` is the one that
produced them.

**Model (Hugging Face):** <https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw>
(public, gated: accept the terms once, then use a token).

Base model: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>

The pack holds SAGE MixedK EXL3 routed experts (mixed K1-K4 per projection,
1.62 bpw), EXL3 MTP draft experts, an EXL3 K8 output head, native block-32 FP8
for every other linear, and the Engram n-gram tables. The Engram tables are
streamed from NVMe; everything else is meant to be device-resident. Under vLLM
on one Spark that does not quite fit, so this recipe serves it with a
**cold-expert alias side file**: the 1,380 least-routed experts plus the
embedding are gathered into one 64-byte-aligned file that the GPU reads through
a read-only mapping, which leaves every hot expert resident with ~4 GiB to spare.

## Status

| Deployment | Folder | Model | Engine | Status |
|---|---|---|---|---|
| 1x Spark, TP1 | `scripts/` | vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw | spark-ds41f r6w + vllm-exl3 + exllamav3 (tile-K 16) + shim, DSpark k=5 | **Measured, serving** |

## Quick start (one Spark)

```bash
export HF_TOKEN=hf_...               # account must have accepted the model terms
bash scripts/download_and_verify.sh  # hf_xet download at the pinned revision + sha256 verify
bash scripts/build_image.sh          # base engine image + our layer (see below)
bash scripts/build_cold_alias.sh     # 8.59 GiB side file, read-only on the pack
bash scripts/serve.sh                # detached; arms the memory guard
bash scripts/guard.sh status         # container + endpoint + memory snapshot
bash scripts/smoke.sh                # plain completion, must contain "Paris"
```

The API is OpenAI-compatible at `http://<host>:8000/v1`, model id `dsv41`.
`DRY_RUN=1 bash scripts/serve.sh` prints the full launch configuration without
touching the box.

## The measured flag set (do not deviate without re-measuring)

The engine is a derived image: `dsv41-1017-ds41f:r6w` (Christopher Owen's
spark-ds41f r6w, upstreams pinned at `2dc95329`) plus our two forks plus a
quant-config shim. Nothing in the base image is replaced.

| Flag | Why it is there |
|---|---|
| `MOE_TILESIZE_K=16` (compile-time) | the served MoE kernel instance; 32 (upstream) is 25 % slower per step |
| `--quantization exl3_dsv41` + `--hf-overrides {...}` | EXL3 routed experts with the engine's native block-32 FP8 for every non-routed weight; the override restates it because the DSpark drafter re-derives the config |
| `--speculative-config dspark k=5` | k=5 is the step/token knee at this step time; k=7 costs +17 ms for +0.55 tok/step |
| `VLLM_EXL3_MIXEDK_MODE=unified` | one kernel for all K groups instead of one dispatch per group (+9 % mean) |
| `VLLM_DS41_MARKOV_NVFP4=1` | the engine's DSpark Markov draft block in NVFP4; measured structured-prompt gain (7.6 -> 11.2 tok/s at the time) |
| `VLLM_DS41_L2_PREFETCH=0` | prefetch measured 8.7 ms/step of work that hid nothing |
| `VLLM_EXL3_ATS_ALIAS=1` + `_LAYERS=38,39` + `_MTP=1` + `_EMBED=1` | ATS aliasing on, with the per-expert manifest in `/alias`. When the manifest lists experts (the cold set does), it decides membership and the layer list is ignored; the values are kept as measured |
| `--compilation-config cudagraph_mode=FULL_DECODE_ONLY`, sizes 1..7 | decode runs as full CUDA graphs (measured +11-14 % over eager) |
| `--max-num-batched-tokens 2112 --long-prefill-token-threshold 2047` | the aligned-prefill chunk; 8192 does not fit |
| `--kv-cache-memory-bytes 1073741824` | 1 GiB KV = 255,211 tokens at the engine's `auto` (bf16) cache dtype |
| `--no-enable-prefix-caching` | measured off |
| `B12X_AUTOTUNE=0` | the base image's optional per-kernel autotune sweep is hours; prebuilt artifacts are the measured path |
| `--security-opt seccomp=unconfined --security-opt label=disable --ulimit memlock=-1` | the Engram reader initialises io_uring; a default seccomp profile aborts it |
| `--shm-size=2g` | measured value; the engine's shared-memory use is small at `--max-num-seqs 1` |

Host side, while serving: `vm.compaction_proactiveness=0`,
`vm.watermark_boost_factor=0`, `vm.swappiness=0`, swap reset before boot.
`scripts/guard.sh stop` restores them.

## Measured bring-up (single Spark)

| Phase | Measured |
|---|---|
| Weights routed | 118 s |
| Engine init | 13.9 s |
| Graph capture | ~1 s |
| **API ready (container start to `/v1/models` 200)** | **3 min 11 s** |

Re-boots are faster once the JIT caches under `~/.cache/dsv41-ds41f/kkref/jit`
are warm; the first boot after an image build pays the TileLang/B12X compile
once.

## Measured memory (121.7 GiB unified)

| Metric | Value |
|---|---|
| Engine reservation (`cuda_alloc` at ready) | 110.5 GiB |
| Steady-state MemAvailable | ~4.0 GiB |
| Swap | 0 KiB growth, 0 kills |
| Cold-expert alias side file | 9,228,943,360 B (8.59 GiB) on disk, mapped read-only |
| KV pool | 1 GiB = 255,211 tokens |

All-experts-resident does not fit under vLLM on this part: un-aliasing layers
38 and 39 alone costs 4.6 GiB of transient load pressure and the guard kills the
boot. That is the whole reason for the side file. Guard policy: 0.5 GiB
MemAvailable floor, 64 MiB swap-growth allowance, 0.05 s sampling
(`scripts/guard.sh`). Do not serve without it armed.

## Measured speed

One row per configuration. `bench7.py`, single stream, 512 generated tokens with
`ignore_eos`, three unique prompts per class (prose / code / structured).
`ms/step` and `tok/step` are read from the engine's own spec-decode counters;
`tok/s` is client-observed. Each row is one boot; this box's acceptance varies
run to run, so compare `ms/step` first.

| Configuration | ms/step | accepted tok/step | tok/s |
|---|---:|---:|---:|
| No draft (cold alias, pre-tile-K kernel) | 74.4 (ms/token) | 1.00 | 13.4 |
| DSpark k=6, grouped MoE, whole-layer alias (38/39 + MTP) | 296-427 | 4.7-5.7 | 13.5 |
| + cold-expert ATS alias | 202.8 | 5.02 | 24.2 |
| + second kernel variant (k4) | 188.8-191.2 | 5.00 | 25.7-27.5 |
| + tile-K 16 build, k=5 (serving) | 140.0-143.5 | 4.35-4.67 | 31.3-34.0 |

Serving configuration per class, two ship boots: prose 137.6-138.4 ms/step,
code 143.2-146.2, structured 142.2-144.5.

Where the step time goes at tile-K 16 (nsys, 974 decode steps, 2,657 kernels/step):
MoE 84.0 ms (58 %), non-MoE linears 36.6 ms, elementwise glue 8.5 ms, head
5.7 ms, draft MoE 6.1 ms. The MoE kernel runs at 1 CTA/SM, 8 warps/SM and 47.6 %
of DRAM peak; it is bound by instruction issue in the trellis decode (about four
integer instructions per weight), not by memory bandwidth.

## Measured prefill

Repeated-text ladder on the earlier (pre-tile-K) kernel (`longprompt.py`, plain
completions, `max_tokens 8`):

| Prompt tokens | TTFT | Prefill tok/s |
|---|---:|---:|
| 1,907 | 1.931 s | 988 |
| 7,631 | 6.586 s | 1,159 |
| 27,949 | 22.556 s | 1,239 |

| Varied-text prefill on the current configuration | TODO(confirm) |
|---|---|

The varied-text ladder (`bench/prefill_varied_probe.py`, python stdlib source,
2K/8K/16K/30K) is being measured on the serving configuration; the row above is
deliberately empty until it lands. The repeated-text figures are not a
substitute: a repeated sentence routes to few experts, so they likely over-read
what real prompts get.

## Quality and correctness

- **Tile-K 16 numerics versus a true fp32 reference:** rel-L2 2.491e-4 (up) /
  2.188e-4 (down) against a 0.50-0.51 fp16 ulp floor, uniform in K and N. The
  build is not one ulp worse than the base kernel anywhere; the base-vs-tile-K
  delta is two independent realizations of the same 0.5-ulp C-buffer rounding.
- **Teacher-forced logprob checks** on every shipped arm (32 positions, full
  vocabulary): top-1 agreement 78-97 % against the non-speculative reference
  depending on the arm, max |dlogprob| inside that session's own noise floor.
  The gate exists to catch a wrong kernel, not to score quality.
- **Greedy output is not bit-deterministic** on this engine: batched atomics in
  the MoE reduce reorder sums. Two runs of the same prompt can differ in the
  tail. Do not build exact-match tests on it; compare logprobs.
- Pack-level quality figures are on the model card. This recipe did not re-run
  them.

## Known limits

- **50-60 tok/s is not reached.** 31-34 tok/s is the best measured so far;
  kernel work continues on the fork branch. Getting to ~46 tok/s needs ~42 ms out of a 140 ms step; the
  ranked candidates and the arithmetic are in `docs/ENGINEERING_NOTES.md`.
- **Vision is not enabled** (`--language-model-only`); no image path was measured.
- **Chat endpoint on a fresh download: TODO(confirm).** The measured pack
  directory also held a `chat_template.jinja` carried over from an earlier upload
  of this pack; the current Hub revision does not ship one. All speed figures use
  plain `/v1/completions`; `/v1/chat/completions` against a fresh download has not
  been checked.
- **Image build: TODO(confirm).** The measured server bind-mounted the same fork and
  plugin trees (identical file hashes to the pinned commits) into the engine image;
  the baked build in `docker/Dockerfile` has not yet been run end to end.
- **`--max-num-batched-tokens 8192` does not fit.** 2112 is the measured value.
- **32K context.** `max_position_embeddings` is 1M and the KV pool is sized for
  255K tokens, but only 32K was measured; longer needs its own ladder.

## Do not

- Do not serve without `scripts/guard.sh` armed (0.5 GiB floor, 64 MiB swap).
- Do not drop the cold-expert alias (`VLLM_EXL3_ATS_ALIAS=0`) and expect a boot:
  the pack does not fit resident.
- Do not raise `--max-num-batched-tokens` to 8192 without re-measuring; it OOMs.
- Do not build the fork without `MOE_TILESIZE_K=16`; you get the 25 %-slower
  kernel and none of these numbers apply.
- Do not enable prefix caching (measured off) or `B12X_AUTOTUNE` (hours of
  bring-up) as a "helpful" default.
- Do not run the container without `seccomp=unconfined`; io_uring init fails.
- Do not print or commit `HF_TOKEN`.

## Cards tested on

| Card | VRAM | Status |
|---|---|---|
| NVIDIA DGX Spark (GB10) | 128 GB unified | Measured (this recipe) |

Nothing else has been verified. The recipe needs ~110.5 GiB of device
reservation plus ~4 GiB of headroom, and depends on unified memory: the alias
side file only works because device and host memory are the same pool.

## Pack revision

Measured against revision `7cdebca3f8a348e423ad293a2ea5cca2297ace1f`. The Hub's
`main` is now `04f7138b8ce3c82399eb4900d8bad12e40a0a0da`, which is what
`scripts/env.sh` pins. `7cdebca3` no longer resolves through the Hub API: the
repo's history is a single `Initial commit` at `04f7138b`, so the earlier
revision was replaced rather than advanced. The check therefore compares content:
`04f7138b` returns the same 21 files with byte-identical sizes to the measured
copy (17 shards, index, `config.json`, `tokenizer.json`, `tokenizer_config.json`),
so it carries no weight or config change relative to the measurement.
`scripts/SHA256SUMS` is the Hub's own LFS sha256 for the 18 LFS files plus
sha256 of the three small ones.

## Credits and license

- Engine: [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f)
  release r6w (Apache-2.0): vLLM fork, TileLang SM121 kernels, DeepSeek
  TileKernels, sparknet. The base image is built per
  [justincasey/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark](https://github.com/justincasey/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark)
  `docker/Dockerfile`.
- Runtime forks pinned here: `vcruz305/exllamav3` (mixed-K unified MoE kernels,
  aarch64 guards) and `vcruz305/vllm-exl3` (EXL3 quant method, ATS loader).
- Model: DeepSeek-V4.1-Flash-1017 (MIT), quantized by
  [vcruz305](https://huggingface.co/vcruz305).
- Recipe scripts: MIT.

## Repo layout

| Path | What |
|---|---|
| `scripts/env.sh` | pins (model revision, image, fork/plugin SHAs), paths, serve knobs |
| `scripts/download_and_verify.sh` | hf_xet download at the pinned revision + sha256 verification |
| `scripts/build_image.sh` | base engine + forks + shim image build |
| `docker/Dockerfile` | the layer: vllm-exl3, exllamav3 at tile-K 16, shim |
| `scripts/build_cold_alias.sh` | builds the cold-expert side file from your own pack copy |
| `tools/ats_build_alias.py` | the gatherer (64-byte alignment, manifest, selection) |
| `configs/cold-alias-selection.json` | the measured 1,380-expert membership |
| `scripts/serve.sh` | host wrapper: docker run, sysctls, memory guard |
| `scripts/serve_inner.sh` | the exact measured in-container vLLM invocation |
| `scripts/guard.sh` | memory guard, status, stop (restores sysctls) |
| `scripts/smoke.sh` | plain completion check |
| `scripts/SHA256SUMS` | per-file sha256 for the pinned pack revision |
| `shim/` | `dsv41_exl3_shim`: quant-config name shim, no vLLM source changes |
| `bench/bench7.py` | ms/step, accepted tok/step, tok/s per class |
| `bench/longprompt.py` | repeated-text prefill ladder |
| `bench/prefill_varied_probe.py` | varied-text prefill ladder |
| `bench/compare_tf.py` | teacher-forced top-1 / dlogprob comparison |
| `docs/ENGINEERING_NOTES.md` | what was tried, what won, and what lost |
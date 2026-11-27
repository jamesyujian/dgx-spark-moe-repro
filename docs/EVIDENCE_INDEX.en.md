# Evidence index: each advisory conclusion → how this kit reproduces it / where the raw artifact lives

*English version of `docs/EVIDENCE_INDEX.md`. The Chinese file remains the reference.*

> **Canonical copy (2026-11-25).** This file and `EVIDENCE_INDEX.md` form the single bilingual canonical pair (digits identical, Chinese wins on wording). Five other same-named copies exist locally; all were **banner-marked and set read-only on 2026-11-25** (see §6) and must never be cited. All future updates go to this pair only.

> **Pinned numbers (frozen 2026-11-25).** These are the sole external figures. If any document disagrees, **that document is wrong — fix the document, not this table.** Changing a row requires a new raw-artifact path and a new evidence grade.

| Quantity | Canonical value | Citation constraint |
|---|---|---|
| 2-node KV pool | **243,733 tokens @ 8,192 = 29.75×** | 249,232 is a ±2 % same-operating-point note only; the 19.65 GiB / 2,688,671 line is retracted; **never write "declared ceiling"** |
| 115 NVFP4 vs BF16 | **1.56×** (TP=2 · c4, 472.76 → 739.68) | **8.28× is permanently void**; always carry "TP=2 / c4" |
| 115 TP scaling | **1.13×** (c4, 654.3 → 739.68) | Single-node PCIe basis only |
| 101 / 102 NVFP4 vs BF16 | **2.47× / 2.37×** | Single-node TP=1 basis |
| 115 concurrency | c1 **250.8** / c2 357.4 / c4 654.3 / c8 **1,017.2** / c12 1,314.02 / c16 **1,551.75** | Whole-segment means only; never cite a single window |
| 101 concurrency | c1 **81.8** / c2 124.7 / c4 **188.2** / c8 267.9 | c4 is the 24 h reference; c8 multiplier has fallen to 1.42× |
| Energy efficiency | **6.7** (GB10, GPU domain) / **3.7** (PRO 5000 c8, full-card TBP) / **5.19** (c16, full-card TBP) | 6.7 **is not** system efficiency; all three denominators must appear together |
| Zero decay | **+0.37 / +0.16 / 0.00 %** (101 / 102 / 115) | First 1 h → last 3 h; 24.00 h, errs 0 |
| Decode engine delta | at 8k SGLang leads marlin by **15.3–16.0 %**, b12x by **20.5–21.2 %**; at 16k/32k/64k it leads marlin by **17.3 / 19.6 / 24.6 %** | Median of 5 reps; the baseline must be named |
| Network | **27.7 GB/s** (bidirectional aggregate; 25 GB/s one-way line rate); **NVLink = 1/14** | Both need their basis/reference stated |
| Model size | **155.4 GiB** / 165.27 B params / **7.63 bit-per-param** | **Never write "156 GB original"** — that 156 GB *is* the official mixed-quantized release; the ≈330 GB original is a projection only |
| Capability gate | **155.4 GiB > 121 G usable on one node (+34.4 GiB)** | Never write "fp8 at ~78 G would fit" — the arithmetic premise is void |
| Community figures | 4-node ring single stream **90** / c16 **325** | Single self-media run, no re-check, no log — **all three caveats required** |
| 116 | **Freezes after 13.84 h, mechanism undetermined** | Never write "suspected forward compat" or "driver lag"; 6 h does not cover the freeze point |

> Path convention: `$WORK` = this kit's root; `$BASE` = the original experiment directory (the root used when these experiments ran; replace it with your own layout).
> Principle: **no conclusion enters the body without a reproducible artifact.**

## 1. Precision conclusions

| Advisory item | Conclusion | How to reproduce | Raw artifact |
|---|---|---|---|
| §1.1 L1 banned | non-ue8m0 fed straight into DeepGEMM → hard crash (SM120 `gemm.hpp:346`; SM121 gives `:415` on `0.2.3.post2`), not silent NaN | **run it from this kit**: `python3 scripts/s104_semantic.py` (needs DeepGEMM + a non-ue8m0 FP8 checkpoint; line numbers move with the version) | `data/reference_l1_semantic/s104_result_sm120.json` (SM120 / 2.8.1, last one `error`) + `s104_result_sm121_dg023post2.json` (SM121 rerun, last one `error`; provenance and sha256 in `README.txt` beside them) |
| §1 FP8 L2 usable | block-scaled 128×128 W8A8 runs | `scripts/run_matrix.sh $MODEL 8230 fp8_tp2 --tensor-parallel-size 2 --moe-backend triton` | `$BASE/s104b/results/s104b_fp8_tp2_{triton,tc}.json` |
| §1 L3 three backends refuse | `hpc` / `cutlass` / `flashinfer_cutlass` each refuse | same as above, change `--moe-backend` | `$BASE/logs/fp8_matrix/{single_hpc,single_cutlass,p3_single_flashinfer_cutlass}.log` |
| §1 GSM8K gap −0.38 pp | indistinguishable at same version and same TP | `scripts/eval_gsm8k.py` twice (one BF16, one FP8) | `$BASE/results/gsm8k_bf16_run2.jsonl`, `$BASE/s104b/results/gsm8k_fp8_tp2_triton.jsonl` |
| §5 NVFP4 not reproducible over ten rounds | kernel non-determinism | ten rounds of `probe_20_prompts.py` + `attrib_tp_vs_dtype.py` | `$BASE/d4/results/s104i_base_marlin_r{1..10}.json`, `s104i_base_humming_r{1..10}.json` |
| §5.3 TP attribution | TP is not the source; the kernel is | ten rounds each at TP=1 / TP=2 + ten rounds BF16@TP2 | `$BASE/d4/results/{s104h_nvfp4_tp2,s104i_base_marlin,s104k_bf16}_r{1..10}.json` |
| §10 tool calling | `qwen3_xml` is mandatory | `probe_20_prompts.py`, three rounds × 2 stacks | `$BASE/d4/results/a10e_*_r{1,2,3}.json` |

## 2. Backend availability conclusions (§4.1 matrix)

| Cell | How to run | Artifact |
|---|---|---|
| 115 / 48 G / 0.29.0 | `run_matrix.sh` across 9 backends, once each | `$BASE/logs3/*_serve.log`, `$BASE/results3/manifest.csv` |
| 116 / 32 G / 0.31.0 | container method (see `env/docker.md` §3), once each | `$BASE/logs/probe2610_*_full.log` |
| 101/102 / GB10 / 0.27.1 | container method, once each | `$BASE/logs/probe_116_2610*.log` (same script family) |
| **cutedsl@SM120** | single container + `--moe-backend flashinfer_cutedsl` | `$BASE/logs/a7_cutedsl_sm120{,_container}.log` |

## 3. Communication and concurrency

| Conclusion | How to run (exact command) | Artifact |
|---|---|---|
| all_reduce 4K 9.8 µs / 32K 16.0 / 128K 45.7 / 512K 80.8 | `torch.distributed.run --nproc_per_node=2 scripts/allreduce_lat.py` (**must be tee'd to disk**) | `$BASE/logs/fp8_matrix/allreduce_w2.txt` |
| concurrency curve 209 → 1011 tok/s (c=1…24) | `vllm bench serve --backend openai-chat --base-url http://127.0.0.1:<P> --endpoint /v1/chat/completions --model <M> --tokenizer <MODEL_DIR> --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 32 --max-concurrency <C> --ignore-eos --seed <S>` | `$BASE/d4/logs/a9_concurrency_sweep_v{2,3}.log` |

> ⚠ Two bench pitfalls (learned the hard way): (1) without a local `--tokenizer` path, an offline machine hangs on HF fetch timeouts; (2) piping the output lets grep swallow the metric lines — always write to a file and grep afterwards.

## 4. Numbers that appear in reports but are **not produced by this kit**

| Number | Why it is not in this kit |
|---|---|
| SGLang's −6.31 % throughput (denominator = `marlin`; the reverse reading is +6.74 %) | the raw artifacts are twelve files under 101 `$BASE/sm12x/results/` (`sglang_result_*.json` + `gsm8k_*.jsonl`, all names in N1 §6), but the **paired executor is not shipped**, so results can be re-checked but not re-run in one command |
| The 5080 node's 18 % drift in 3 minutes | a different GPU model, noise evidence rather than a conclusion of this kit |
| `flashinfer_cutile` capacity numbers (16/48 MiB) | needs a single run of that backend on a 32 G card; numbers are in the advisory body, scripts as in §2 |


## 5. Evidence map for N2 (the GB10 dual-node MoE recipe)

> Full English edition: `docs/N2_dual_node_recipe_v0.1.en.md` (numbers match this section digit for digit; the Chinese text wins).

| Claim | Where in the kit | Cross-check / boundary |
|---|---|---|
| PP=2 works (NGC 26.08 + Ray + `--enforce-eager`); 270 s to serve, health 200 | `docs/N2_GB10双机MoE配方_v0.1.md` §3–§4 plus the `pp2_ngc_entry.sh` text in §14.2 | Container run flags in §14.1 (`--gpus all` **and** `--ipc=host`, either missing kills startup); raw logs (48–59 KB) are not shipped — fetch per §10 on the machines |
| Single stream **14.71 / 14.72 t/s** (120 tok / 8.16 s); cold first request 4.4 discarded | §5.1 | Probe `pp2_e2e.py`; three logs `pp2_e2e_r{1,2,3}` |
| 4-way concurrency **range 25–43 t/s** (25.5 / 43.2, `max_tokens 300`) | §5.1 | **No point estimate is publishable**; the 32.6 reading from 12:35 has no log and is history only |
| KV **243,733 tokens / 29.75×**, PP0 24.32 / PP1 25.01 GiB | §5.1 + startup log | The 249,232 / 30.42× pair is the same operating point ±2%; the 19.65 GiB / 2,688,671 line was **retracted** (not reproducible) |
| Cross-node **TP=2 works** (anemll 0.1.1 + `mp`, `:8888`); the long-context sweep ran on it | §6 plus §14.3/§14.4 | Receipt `tp2s.out` (HEALTH_OK 266 s); sweep source `d8a_longctx.json` recomputes cell by cell |
| SGLang prefill **4,468 / 4,597 t/s**, decode **31.5 / 31.7 t/s** | §7 | Re-measured with logging at 13:39; warning text `Config file not found … NVIDIA_GB10.json`; older console figures 4,534–4,648 are not citable |
| llama.cpp pp512 **831.78 ± 12.07** / tg128 **12.46 ± 0.05** | §7 | Reference stack only; scripts and weights are not shipped |
| NVFP4 large shapes **3.6–4.3×**, small shapes mostly parity (one 3.28× exception), error 13.9–16.7%, FP8 column `Unknown recipe` | §8 | Needs `$BASE/envs/sm120-t211` (torch 2.14.0+cu130) and `cd $BASE/src/DeepGEMM-vllm` first |
| anemll 0.1.1 PP=2 is dead (`stride 16`) | §9 | **No log, and cannot be re-run** — the `dsv4-abliterated` weights no longer exist on either node |

## 6. Same-named files and freeze record (2026-11-25)

Inventory: 7 `EVIDENCE_INDEX*.md` files locally. **Canonical set = 2 (this file + the Chinese file, updated and released together)**; the other 5 are historical copies, banner-marked and set `IsReadOnly = True`.

| # | Path | Disposition | Basis |
|---|---|---|---|
| 1 | `2026-11-20\new-chat\sm12x-repro-release\docs\EVIDENCE_INDEX.md` | **Canonical (Chinese)** | 62 lines, 2026-11-22 13:48, carries the 2-node KV and all latest values |
| 2 | `EVIDENCE_INDEX.en.md` (same dir) | **Canonical (English)** | Digit-identical to #1, released together |
| 3 | `2026-11-18\new-chat-1\sm12x-repro\docs\EVIDENCE_INDEX.md` | Frozen | 41 lines, 11-19 09:55, oldest |
| 4 | `2026-11-20\new-chat\_verify106\ext\dgx-spark-moe-repro\docs\EVIDENCE_INDEX.md` | Frozen | 44 lines, 11-20 09:40 |
| 5 | same dir, `EVIDENCE_INDEX.en.md` | Frozen | same |
| 6 | `2026-11-20\new-chat\_verify107\dgx-spark-moe-repro\docs\EVIDENCE_INDEX.md` | Frozen | 44 lines, 11-20 10:07 |
| 7 | same dir, `EVIDENCE_INDEX.en.md` | Frozen | same |

**Freeze criterion**: copies ≤44 lines and lacking `243,733` (i.e. missing the 2-node KV and the whole of §5), so they cannot coexist with the canonical set as a citation source. **Unfreeze procedure**: clear `IsReadOnly`, drop the banner, log purpose and date in this table — never edit numbers directly.


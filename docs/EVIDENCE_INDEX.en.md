# Evidence index: each advisory conclusion → how this kit reproduces it / where the raw artifact lives

*English version of `docs/EVIDENCE_INDEX.md`. The Chinese file remains the reference.*

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


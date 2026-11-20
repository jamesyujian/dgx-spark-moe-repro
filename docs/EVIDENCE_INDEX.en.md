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
| SGLang's −6.31 % throughput (denominator = `marlin`; the reverse reading is +6.74 %) | the raw artifacts are twelve files under 101 `~/sm12x/results/` (`sglang_result_*.json` + `gsm8k_*.jsonl`, all names in N1 §6), but the **paired executor is not shipped**, so results can be re-checked but not re-run in one command |
| The 5080 node's 18 % drift in 3 minutes | a different GPU model, noise evidence rather than a conclusion of this kit |
| `flashinfer_cutile` capacity numbers (16/48 MiB) | needs a single run of that backend on a 32 G card; numbers are in the advisory body, scripts as in §2 |

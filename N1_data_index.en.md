# N1 data index

*English version of `N1_data_index.md`. Same rows, same status marks; where wording differs, the Chinese file wins.*

> Path convention: `$BASE` = the original experiment root (Linux side); files inside this kit are given relative to the repo root. One row per conclusion. No explanation, no recommendation.

| # | Conclusion (N1 item) | Raw file | Conditions | Status |
|---|---|---|---|---|
| 1 | §1.1 L1 · SM120 hard failure (`gemm.hpp:346` / `CUDA 719`) | `$BASE/s104_semantic.py` (8 cases; body not found); log excerpts in `安全公告_SM12x非ue8m0_FP8静默NaN_2026-11-13.md` | 116 / PRO 4500 32G / DeepGEMM 2.8.1 direct / `disable_ue8m0_cast=True` | ⛔ rechecked 2026-11-20: script not found on 101 or 116 either; only the document excerpt survives |
| 2 | §1.1 L1 · SM121 silently all-NaN (`exit=0`, no warning) | 101/102 `$BASE/logs/` (same script, SM121 side); excerpts in the same document | GB10 / SM121 / same script / subprocess isolation | ⛔ 101/102 are back up; no SM121-side log for this script was persisted; excerpt only |
| 3 | §1.2 L2 · TP1 34/40 | `$BASE/s104b/results/s104b_fp8_single_triton.json` | 115 / 0.29.0 / PRO 5000 48G / TP=1 / `--moe-backend triton` | ○ on 115 |
| 4 | §1.2 L2 · TP2 36/40, failure set identical to BF16 | `s104b_fp8_tp2_triton.json`; reference `s104b_bf16.json` (TP2, 11-15) | 115 / 0.29.0 / TP=2 / triton | ○ on 115 |
| 5 | §1.2 L2 · GSM8K 0.8453 vs BF16 0.8491, −0.38 pp, p=0.63 | `s104b/results/gsm8k_fp8_tp2_triton.jsonl`; BF16 side `$BASE/results/gsm8k_bf16_run2.jsonl` (same machine, 115) | same version and TP / 1319 items / per-item pairing | ○ on 115 (both sides) |
| 6 | §1.3 L3 · `hpc` / `cutlass` / `flashinfer_cutlass` all refuse | `$BASE/logs/fp8_matrix/{single_hpc,single_cutlass,p3_single_flashinfer_cutlass}.log` | 115 / 0.29.0 / TP=1 or 2 / switching `--moe-backend` | ○ on 115 |
| 7 | §4.1 · 115 / 48G / 0.29.0 nine-backend matrix | `$BASE/results3/manifest.csv`, `$BASE/logs3/*_serve.log` (27 logs) | 115 / 0.29.0 / NVFP4 / 9 backends once each | ○ on 115 |
| 8 | §4.1 · 116 / 32G / 0.31.0 per backend | `$BASE/logs/probe2610_*_full.log` (b12x / cutlass / vllm_cutlass etc.) | 116 / 0.31.0 / container method | ○ on 116 |
| 9 | §4.1 · 101/102 GB10 0.27.1 per backend | `$BASE/d4/logs/d4r*_sm121_*.log`, `$BASE/d4/results/d4r_nvfp4_*_sm121_*.json` | GB10 / 0.27.1 / container method | ○ on 101/102 (15 `d4r*_sm121_*.log`, 2 `d4r_nvfp4_*_sm121_*.json` per machine, confirmed) |
| 10 | §4.1 · `cutedsl@SM120` device-level refusal (0.31.0, 40 s) | `$BASE/logs/a7_cutedsl_sm120{,_container}.log` | 116 / 0.31.0 / `--moe-backend flashinfer_cutedsl` | ○ on 116 |
| 11 | §5.1 · 115 `marlin` 10 rounds: lp-identical 0/20, Δlp **1.263** | `$BASE/d4/results/s104i_base_marlin_r1..10.json` | 115 / 0.29.0 / 48G / TP=2 / temp=0 / seed=0 / same process | ✔ recomputed 2026-11-20 |
| 12 | §5.1 · 115 `humming` 10 rounds: 0/20, Δlp **1.529** | `s104i_base_humming_r1..10.json` | same, backend switched | ✔ recomputed |
| 13 | §5.1 · 116 `marlin` 10 rounds: 1/20, Δlp **1.899** | `$BASE/d4/results/s104j_116_marlin_r1..10.json` | 116 / 0.31.0 / 32G / `marlin` | ✔ recomputed (text / `num_unit` / lp) |
| 14 | §5.1 · 116 `humming` 10 rounds: 0/20, Δlp **1.897** | `s104j_116_humming_r1..10.json` | 116 / 0.31.0 / 32G / `humming` | ✔ recomputed |
| 15 | §5.1 · BF16 TP=2 ten rounds 20/20, Δlp **0.000** | `$BASE/d4/results/s104k_bf16_r1..10.json` | 115 / 0.29.0 / TP=2 / same 20 prompts | ✔ recomputed (Δlp exactly 0.0000) |
| 16 | §5.1 · SM121 four stacks 0/80, Δlp 1.62–2.09 | 101 `$BASE/d4/results/s104x3_det_{marlin,flashinfer_b12x}_sm121_101_r1..10.json` | GB10 / 0.27.1 / `marlin` and `b12x`, 10 rounds each | ○ on 101 (r1–r10 confirmed) |
| 17 | §5.2 D · Δlp rises after `--enforce-eager` (1.258 → **1.905**, 3 vs 3 rounds, same stack) | `s104f_det_humming_r1..3.json`, `s104f_eager_humming_r1..3.json` | 115 / 0.29.0 / `humming` / `--enforce-eager` + `--max-num-seqs 1` + `--max-num-batched-tokens 2048` | ✔ recomputed (+0.647; 3-round subset SD 0.071) |
| 18 | §5.2 C · `VLLM_BATCH_INVARIANT=1` fails at serve time | `$BASE/d4/logs/` (serve failure); excerpt in `2026-11-16_D4_执行记录.md` | 115 / 0.29.0 / NVFP4 W4A16 | ○ |
| 19 | §5.3 · 13 stable prompts (text) × 4 stacks × 10 rounds = 40 runs | the four files in rows 11–14 (`s104i_*`, `s104j_*`) | 115/116 / 0.29.0, 0.31.0 / `marlin`, `humming` / 10 rounds each | ✔ recomputed (13/20 text-constant; `zh_en` flips only on 116 `humming`, 2 variants) |
| 20 | §5.3 · TP attribution: TP1 flips 5/20, TP2 6/20, BF16@TP2 0/20 | `s104i_base_marlin_r1..10` (TP1), `s104h_nvfp4_tp2_r1..10` (TP2), `s104k_bf16_r1..10` | 115 / 0.29.0 / ctx 2048 / util 0.90 | ✔ recomputed (TP2 side: `num_unit` shows 3 values over 10 rounds) |
| 21 | §5.3 · BF16@TP=1 physically unreachable (71.9 GB > 47.8 GiB) | no raw file; `du -sb` and `nvidia-smi` output recorded in the 2026-11-19 notes | 115 / BF16 weights 71,926,861,539 B / single card 48,935 MiB | ⛔ physically unreachable |
| 22 | §5.4 · `num_unit` 116 `marlin` 1/10, `humming` 8/10 | `s104j_116_{marlin,humming}_r1..10.json` | 116 / 0.31.0 / 32G / `seed=1234` / `max_tokens=40` | ✔ recomputed (1/10, 8/10) |
| 23 | §5.4 · `num_unit` 3 processes × 10 calls: 4 / 6 / 0 | `$BASE/d4/results/num_unit_cycles.jsonl` | 116 / 0.31.0 / `marlin` / three separate processes | ○ on 116 |
| 24 | §5.4 · 115 three stacks `num_unit`: `marlin` 10/10, `humming` 9/10, BF16 10/10 | `s104i_base_{marlin,humming}_r1..10`, `s104k_bf16_r1..10` | 115 / 0.29.0 | ✔ recomputed (10 / 9 / 10; answer text always `3584`, lps still not bit-identical) |
| 25 | §5.5 · four groups, 5 rounds each (1098…1111 correct) | 115 `$BASE/d4/results/gsm8k_0{29,31}_r*.jsonl`; 116 `gsm8k_116{mar,hum}_r{1..5}.jsonl` | four stacks, 5 rounds × 1319 each | ○ on 115 / 116 (this round only r4–r5 used, for `p_d`) |
| 26 | §5.5 · resolution `p_d ≈ 4.7 %` → threshold 1.17 pp | `gsm8k_031_r4.jsonl` vs `r5.jsonl` (b=29 / c=33 = 62/1319) | 115 / 0.31.0 / `marlin` / two runs of the same stack, per-item pairing | ✔ recomputed (0.0470; other three pairs 0.0425 / 0.0478 / 0.0493) |
| 27 | §5.5 · three-version 9-run calibration 83.573 / 83.472 / 83.321 % | `$BASE/results3/gsm8k_{A,B,C}_r{1,2,3}.jsonl` + `manifest.csv` | 115 / 0.27.1, 0.29.0, 0.31.0 / 3 runs each | ○ on 115 |
| 28 | §6 · SGLang `marlin` 6.31 % faster than `flashinfer_cutlass` | 101 `$BASE/results/gsm8k_sglang_marlin_r{2,3}.jsonl` + `gsm8k_cutlass101_r{2,3}.jsonl` (plus the baseline pair `gsm8k_sglang_{cutlass,sm121}.jsonl`); 102 `gsm8k_sglang_cutlass_r{2,3}.jsonl` | 101/102 / SGLang 0.5.23 / 3 vs 3 | ○ on 101/102 (files confirmed present only; no r1 on disk, the unnumbered pair is the baseline) |
| 29 | §6 · accuracy side p=0.55 (b=25 / c=20) | same jsonl files (per-item verdict) | 101 / same-stack pairing | ○ on 101/102 (b/c not recomputed) |
| 30 | §10.1 · tool calling requires `qwen3_xml` (tc 3/3) | `$BASE/d4/results/a10e_{fp8xml,nvfp4xml}_r{1,2,3}.json` | 115 / 0.29.0 / FP8 and NVFP4 stacks, 3 rounds each | ○ on 115 |
| 31 | §10.9 · concurrency curve 209 → 1011 tok/s (c=1…24) | `$BASE/d4/logs/a9_concurrency_sweep_v{2,3}.log` | 115 / `vllm bench serve` / input 512, output 128 | ○ on 115 |
| 32 | §3 comms · all_reduce 4K 9.8 µs / 512K 80.8 µs | `$BASE/logs/fp8_matrix/allreduce_w2.txt` | 115 / world=2 / 200 samples per size / NCCL 2.30.7 | ○ on 115 (persisted) |
| 33 | A6 / SE-03 · first token `109` vs `123` (5.38 / 2.88 nats), mechanism = spec-decode | 101 `$BASE/d4/results/se03_tf2_{b12x,dgemm,dgemm_formula}_101.json`; scripts `/tmp/se03_tf2.py`, `iso_probe.py` (`/tmp` unchecked) | 101+102 / DeepSeek-V4-Flash / TP=2 / temp=0 | ○ on 101 (3 json files confirmed) |
| 34 | A8 · GGUF full 1319: Q4 = Q8 = 1128/1319 | 101 `$BASE/results/full_q4ngl99full.jsonl`; 102 `$BASE/results/full_q8ngl99full_102.jsonl` | 101+102 / llama.cpp / `-ngl 99` / ctx 8192 / `max_new_tokens=512` | ○ on 101/102 (both full jsonl confirmed) |
| 35 | C3 · FP8+triton+TP2 unpack-and-rerun 18/20 × 3, md5 `c8db9a77…` | this kit `data/reference_replay_1120/r1..3.json` + `meta.txt`; originals 115 `~/c3verify/pkg/results/` | 115 / 0.29.0 / 2×48G / freshly unpacked copy | ✔ shipped (md5 recomputable) |
| 36 | C3 · NVFP4+marlin+TP1 cross-model first-round md5 `74dc1260…` | this kit `data/reference_crossmachine_nvfp4/{115,116}_r1..3.json` + `meta_*.txt` | 115 (2×48G) / 116 (1×32G) / 0.29.0 / TP=1 | ✔ shipped |
| 37 | C3 · baseline rounds (11-19) 18/20 × 3, KV 165.50× | this kit `data/reference_fp8_tp2/r1..3.json` + `meta.txt` | 115 / 0.29.0 / FP8+triton+TP=2 | ✔ shipped |

## Status marks

| Mark | Meaning |
|---|---|
| ✔ | recomputed 2026-11-20 from the raw files; numbers match the advisory |
| ○ | raw file present on 115 or 116, not recomputed this round |
| ⏳ | raw file should be on 101/102 but was not located in the results tree this round (not counted as verified) |
| ⛔ | physically unreachable / no image / nothing persisted; the reason is stated in the advisory |

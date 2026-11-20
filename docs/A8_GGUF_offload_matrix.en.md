# A8 · GGUF offload-ratio latch matrix (conclusion page)

*English version of `docs/A8_GGUF卸载门栓矩阵.md`. The Chinese file remains the reference.*

- Date: 2026-11-19 (data collected 2026-11-15 13:49–19:56; this round is analysis close-out only)
- Machine: **101 (GB10 / SM121, single machine)**
- Engine: llama.cpp `llama-server`, `version: 0 (unknown) built with GNU 13.3.0 for Linux aarch64`
- Models (GGUF)
  - `Qwen3.6-27B-Q4_K_M.gguf` — 16,817,244,384 B, sha256 `5ed60d0a…392a0`, `n_params 26,895,998,464`, `n_embd 5120`, `n_vocab 248,320`
  - `Qwen3.6-27B-Q8_0.gguf` — 28,595,763,424 B
- Launch parameters: `-ngl <99|32|0> --ctx-size 8192 --host 127.0.0.1`, `n_threads = 20 (n_threads_batch = 20)`, `CUDA : ARCHS = 1210 | USE_GRAPHS = 1 | BLACKWELL_NATIVE_FP4 = 1`
- Evaluation method: `eval_subset.py <port> <tag> 30 results/subset_<tag>.jsonl`, GSM8K **fixed first 30 items** (same grading as `eval_gsm8k.py`, `last_number` + `norm`)
- Matrix: `-ngl` three levels × 2 weights × 2 runs = **12 cells, 30 items each**

## 1. Accuracy (constant across the 12 cells)

| Weight | -ngl | r1 correct | r2 correct | acc |
|---|---|---|---|---|
| Q4_K_M | 99 | 22/30 | 22/30 | **0.7333** |
| Q4_K_M | 32 | 22/30 | 22/30 | **0.7333** |
| Q4_K_M | 0 | 22/30 | 22/30 | **0.7333** |
| Q8_0 | 99 | 22/30 | 22/30 | **0.7333** |
| Q8_0 | 32 | 22/30 | 22/30 | **0.7333** |
| Q8_0 | 0 | 22/30 | 22/30 | **0.7333** |

`syntax_valid_rate` is **1.0** in every cell (300/300 generations parsed).

**Conclusion A**: the offload ratio 99 / 32 / 0 has **zero effect** on GSM8K-30 accuracy (constant 22/30); Q4_K_M and Q8_0 are **equal** on this subset (22/30).

## 2. Determinism (6 cells × 2 runs, byte-level)

| Weight | -ngl | per-item pred identical | per-item content identical |
|---|---|---|---|
| Q4_K_M | 99 | **30/30** | **30/30** |
| Q4_K_M | 32 | **30/30** | **30/30** |
| Q4_K_M | 0 | **30/30** | **30/30** |
| Q8_0 | 99 | **30/30** | **30/30** |
| Q8_0 | 32 | **30/30** | **30/30** |
| Q8_0 | 0 | **30/30** | **30/30** |

(content was whitespace-normalised before comparison; nothing else was rewritten.)

**Conclusion B**: with `temp`/`seed` unset, this stack is **byte-identical across repeats within a cell, 30/30**, including `-ngl 0` (pure CPU path) and `-ngl 99` (fully offloaded path).

## 3. Differences between levels (only two, none changing the verdict)

| Weight | run | idx | -ngl 99 | -ngl 32 | -ngl 0 | gold | verdict |
|---|---|---|---|---|---|---|---|
| Q4_K_M | 1, 2 | 26 | `243` correct | `243.00` correct | — | 243 | both correct (pure formatting) |
| Q8_0 | 1, 2 | 8 | `1` wrong | — | `120` wrong | 45 | both wrong (different wrong branch) |

**Conclusion C**: the offload ratio **does not change semantic correctness** — one difference is a trailing-format issue (`.00`), the other is two wrong values swapped (`1` ↔ `120`, both wrong). `ngl99 vs ngl0` on Q4 is 30/30 identical values; `ngl99 vs ngl32` on Q4 is 29/30 (only idx26 formatting).

## 4. Throughput (same 30 items, same ctx)

| Weight | -ngl | r1 tok/s | r2 tok/s | relative to ngl99 |
|---|---|---|---|---|
| Q4_K_M | 99 | 12.46 | 12.49 | 1.00 |
| Q4_K_M | 32 | 5.69 | 5.87 | **0.46×** |
| Q4_K_M | 0 | 4.47 | 4.28 | **0.35×** |
| Q8_0 | 99 | 8.01 | 7.99 | 1.00 |
| Q8_0 | 32 | 4.11 | 4.42 | **0.53×** |
| Q8_0 | 0 | 3.34 | 3.23 | **0.41×** |

elapsed_s (r2 basis): Q4 99 **774.1** / 32 1665.7 / 0 2273.0; Q8 99 1225.9 / 32 2227.8 / 0 3049.0.

**Conclusion D**: the offload ratio directly determines throughput — whatever leaves the GPU runs at CPU speed. Q4_K_M `-ngl 99` is the only fully-offloaded level in this matrix and 12.5 tok/s is its ceiling; `-ngl 0` drops to 0.35×, with Q8_0 structurally similar (0.41×).

## 5. Publication rules (three that must hold)

1. **Never compare against vLLM numbers**: the 4.3–12.5 tok/s here is llama.cpp single-stream (batch=1, no continuous batching) on a 27B GGUF, while 192.27 tok/s for vLLM NVFP4 marlin on 101 is a 35B MoE with a batching stack. The two sets differ in engine, quantization, model and concurrency method.
2. **Accuracy must not be extrapolated**: 22/30 (0.7333) is the GSM8K **first-30** subset and does not represent the full set; the full-set figure is recorded elsewhere in this repository — do not mix citations.
3. **The determinism conclusion covers only this llama.cpp stack**: the 6 cells at 30/30 byte-identical **must not** be used to infer that "NVFP4 is more or less accurate" — vLLM NVFP4 marlin's 6/20 flips are non-determinism (same dtype, same parameters, different answers), a different matter entirely.

## 6. Not evidenced (recorded as-is)

- `grep -i offload` in the serve logs had **no hits**; the actual number of offloaded layers is taken from the `-ngl` argument (99/32/0). No `load_tensors: offloading N/M layers` line was confirmed.
- Only 2 runs per cell, never 4+; concurrency (batch>1) and long-context degradation were not tested.
- Q4_K_M and Q8_0 tying may come from 30-item subset saturation, or because 27B genuinely has no gap at these quantization levels; no full-set verification was done.

## 7. Evidence paths (on 101)

```
$BASE/results/subset_p2_ngl{99,32,0}.jsonl          # Q4_K_M r1
$BASE/results/subset_q4_ngl{99,32,0}_r2.jsonl       # Q4_K_M r2
$BASE/results/subset_q8_ngl{99,32,0}_r{1,2}.jsonl   # Q8_0
$BASE/logs/p2_matrix.log                            # latch-round driver
$BASE/logs/p2_ngl{99,32,0}_args.txt                 # launch parameters (incl. sha256)
$BASE/logs/p2_ngl{99,32,0}_serve.log
$BASE/logs/p2_followup.log                          # determinism round + per-cell summary JSON
```

Recompute:
```bash
python3 /tmp/a8_analyze.py    # accuracy / determinism / inter-level / throughput tables
python3 /tmp/a8_diff.py       # per-item attribution of inter-level differences
```

---

## Revision (2026-11-20, after the full 1319-item run)

The conclusion "accuracy constant across 12 cells at 22/30" **holds only at the total level, not per item**. Full-set evidence:

- Q4_K_M `-ngl 99` full = **1128/1319 = 0.855193**; Q8_0 `-ngl 99` full = **1128/1319 = 0.855193** ⇒ same total, both 191 wrong.
- But per item there are **58 flips (4.40 %) in both directions** (Q4 only 29, Q8 only 29), McNemar p = 1.0000 ⇒ indistinguishable.
- About 3/4 of the flips come from `max_new_tokens=512` truncation (Q4 117 items / 8.9 %, Q8 113 / 8.6 %); excluding them (n = 1179) gives 0.9160 vs 0.9169.
- The 30 subset items overlap the Q4 full run with **30/30 identical verdicts** ⇒ the jitter happens across different items, not within one item.

**Use this wording externally**: "Q4_K_M and Q8_0 have identical accuracy on the full GSM8K set (0.8552), while their per-item error sets differ by 4.40 % two-way flips, attributable to verdict jitter and 512-token truncation rather than to a quantization precision difference."

Details in `docs/A8_GSM8K_full1319_conclusion.en.md` (Chinese: `docs/A8_GSM8K全量1319_结论.md`).

# A8 · GGUF full GSM8K (1319 items) — Qwen3.6-27B Q4_K_M vs Q8_0, `-ngl 99`

*English version of `docs/A8_GSM8K全量1319_结论.md`. The Chinese file remains the reference.*

> 2026-11-20 08:05 | Setup: llama.cpp / single machine, single GPU / ctx 8192 / `n_threads 20` / `enable_thinking=false` / temp=0 / `max_new_tokens=512` / answer = "the last numeric value"
> Two machines in parallel: 101 ran Q4_K_M (finished 23:13), 102 ran Q8_0 (finished 04:25); 1319/1319 with 0 missing, 0 malformed lines, 0 empty content

## 0. One-line conclusion

**Both quantization levels score identically on the full 1319 items (1128/1319 = 0.8552, 191 wrong each)**, yet their per-item error sets differ by **58 items (4.40 %) flipping in both directions** (29 each), McNemar p = 1.0000 ⇒ **Q8_0 is not more accurate than Q4_K_M**. The 58-item difference comes from **run-level verdict jitter plus 512-token truncation**, not from a systematic precision loss caused by quantization.

## 1. Single-run metrics

| Item | Q4_K_M `-ngl 99` (101) | Q8_0 `-ngl 99` (102) |
|---|---|---|
| Items | 1319 / 1319 | 1319 / 1319 |
| Correct / wrong / syntax-invalid | **1128 / 191 / 0** | **1128 / 191 / 0** |
| `gsm8k_accuracy` | **0.855193** | **0.855193** |
| `syntax_valid_rate` | 1.0 | 1.0 |
| Wall clock | **33,490.8 s (9 h 18 m)** | **51,965.8 s (14 h 26 m)** |
| Generated tokens | 417,668 | 418,412 |
| Throughput | **12.47 tok/s** (25.4 s/item) | **8.05 tok/s** (39.4 s/item) |
| `finish_reason` | stop 1202 / **length 117** | stop 1206 / **length 113** |

- Total generated tokens are nearly equal (417,668 vs 418,412, a 0.17 % difference) ⇒ both levels have the same output-length distribution, so the difference is not "Q8 talks more or less".
- Throughput ratio **0.65×** (8.05 / 12.47), matching the 30-item subset (8.01 / 12.46) **exactly** ⇒ subset throughput extrapolates, and the full run gives the first 9–14 hour TCO figure.

## 2. Per-item pairing (all 1319 paired)

| Combination | Items |
|---|---|
| both correct | 1099 |
| both wrong | 162 |
| Q4 only correct | **29** |
| Q8 only correct | **29** |
| **flips total** | **58 (4.40 %)** |

McNemar exact two-sided: discordant = 58, **p = 1.0000** ⇒ indistinguishable, and **symmetric** (29 vs 29) ⇒ each level wins some and loses some; neither is uniformly better.

### Flip mechanism: about 3/4 explained by 512-token truncation

| Group | Q4 truncated inside group | Q8 truncated inside group |
|---|---|---|
| Q4 only correct (29 items) | 12 | **22 (76 %)** |
| Q8 only correct (29 items) | **23 (79 %)** | 9 |

Typical samples (the `classic` vs `length` difference is exactly the verdict difference):

| idx | gold | Q4 pred | Q8 pred | Q4 finish | Q8 finish |
|---|---|---|---|---|---|
| 141 | 400 | **400** | 100 | stop | **length** |
| 283 | 68 | **68** | 8 | stop | **length** |
| 159 | 15 | 1 | **15** | **length** | stop |
| 186 | 5 | 6 | **5** | **length** | stop |
| 27 | 16 | 60 | **16.00** | stop | stop |
| 303 | 1300 | 13 | **1300** | **length** | **length** |

⇒ Whether an item is judged right or wrong depends mainly on "did this trajectory get hard-cut at 512 tokens", not on weight width. Roughly the remaining 1/4 are genuine numeric disagreements (e.g. idx 27: Q4 gives 60, Q8 gives 16).

## 3. Truncation is a methodological defect that must be disclosed

| Level | Truncated items | Share | Correct among truncated | Accuracy among truncated |
|---|---|---|---|---|
| Q4_K_M | 117 | **8.9 %** | 30 | **25.6 %** |
| Q8_0 | 113 | **8.6 %** | 24 | **21.2 %** |

**Sensitivity analysis (dropping items truncated in either level, n = 1179)**:

| Level | Accuracy on untruncated subset |
|---|---|
| Q4_K_M | **0.9160** |
| Q8_0 | **0.9169** |
| difference | **+0.0848 pp** |

⇒ The full-run 0.8552 includes ~8.9 % truncation loss; on clean samples only, both levels sit at **0.916**, differing by **0.08 pp** (far below the ≈1.0 pp single-run standard error for 1319 items).

**Recommended handling**: publish **0.8552 as the official figure** (same method as the existing 12-cell matrix and the N1 citation), and disclose 0.9160 / 0.9169 with the 8.9 % truncation rate as a **sensitivity analysis** side by side; **never** report 0.916 alone as if it were the full set.

## 4. Relationship to the 30-item subset (12 cells) — a method correction

- The 12 subset cells were constant at **22/30 = 0.7333**, and the conclusion at the time was "accuracy is constant across `-ngl 99/32/0`";
- The full run shows **the total is constant** (1128 vs 1128), but **per-item results are not** (58 flips, 4.40 %);
- The 30 subset items overlap the Q4 full run with **30/30 identical verdicts** (same prompts reproduce), so the jitter happens "across different items", not "within the same item".

⇒ Publish this instead of "accuracy constant across 12 cells": **"the total is constant (Q4/Q8 both 1128/1319), while the per-item error sets show 4.40 % two-way flips, i.e. verdict jitter; a 30-item subset cannot resolve that jitter"**. The old "22/30 constant" statement stays as subset evidence but is no longer extrapolated to "per-item constant".

## 5. Citable / not citable

**Citable**
1. Full-set accuracy: Q4_K_M = Q8_0 = **1128/1319 = 0.8552**, syntax-valid rate **1.0**.
2. Throughput: Q4 **12.47 tok/s**, Q8 **8.05 tok/s** (full run 9.3 h / 14.4 h), subset agreeing with the full run.
3. The two levels are **indistinguishable**: McNemar p = 1.0000, symmetric 29/29.
4. Untruncated subset 0.9160 / 0.9169, a 0.08 pp difference.

**Not citable (not evidenced this round)**
1. **Each level was run once**, so the 58 flips cannot rule out "the same configuration also flips on a rerun" ⇒ cannot say "switching levels caused X items to change"; only "the two verdict sets are not identical".
2. The 8.9 % truncation rate is an artifact of `max_new_tokens=512`; **768 / 1024 were not tested**, so we cannot assert "the model itself cannot answer".
3. `-ngl 32/0` was run only on the 30-item subset, **not on the full set** ⇒ the 12-cell conclusion cannot be extrapolated to a full-set per-level conclusion.
4. No repeat run of Q4_K_M at `-ngl 99` ⇒ the magnitude of runtime jitter is unmeasured.

## 6. Evidence paths

| Content | Path |
|---|---|
| Q4 per-item results | 101 `$BASE/results/full_q4ngl99full.jsonl` (1319 lines) |
| Q4 summary | 101 `$BASE/results/full_q4ngl99full_summary.json` |
| Q8 per-item results | 102 `$BASE/results/full_q8ngl99full_102.jsonl` (1319 lines) → copied back to 101 under the same name |
| Q8 summary | 102 `$BASE/results/full_q8ngl99full_102_summary.json` (contains `A8Q8_DONE 04:25:02`) |
| Q8 run log | 102 `$BASE/logs/a8_q8_102.log` |
| Analyzers | 101/102 `$BASE/a8_full_analyze.py`, `a8_pair_detail.py` |
| Generated report | 101 `$BASE/results/a8full/A8_full_report.md` / `.json` |
| Dataset | `$BASE/data/gsm8k_test_1319.jsonl` (1319 items) |

**Recompute (on 101)**

```bash
cd $BASE
python3 a8_full_analyze.py --q4 results/full_q4ngl99full.jsonl \
  --q8 results/full_q8ngl99full_102.jsonl \
  --subset results/subset_p2_ngl99.jsonl --out results/a8full
python3 a8_pair_detail.py     # 58 flips and truncation attribution
```

## 7. Cleanup

- 101's `llama-server` (:30150, pid 1577812, Q4_K_M) **does not exit by itself**; it was cleared with `pkill -f "port 3015[0]"`. Both GPUs back to 0 % util.
- 102's `llama-server` (:30151) needed the same cleanup (until cleared it still held VRAM, without affecting result files).
- 102's results were copied back to 101; `results/a8full/` is the single analysis entry point.

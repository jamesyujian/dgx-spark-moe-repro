# CHANGELOG

*English version of `CHANGELOG.md`. The Chinese file remains the reference; where wording differs, the Chinese text wins.*

## v1.0.7 — 2026-11-20 (SM121 rerun: silent NaN downgraded to a version-scoped claim)

**New artifact**
- `data/reference_l1_semantic/s104_result_sm121_dg023post2.json` — the same `scripts/s104_semantic.py`, run on **101 (GB10 / SM121 / aarch64)** inside `sglang-sm12x:f832b5c39559` (`deep_gemm 0.2.3.post2`, CUDA 13.0.3): 8 `verdict` records, sha256 `126bd77144b0f3d07391d4fde58b8c4582911c9212405d4e5ae5c01d5a15dec3` (1,495 B).
- `data/reference_l1_semantic/README.txt` gains a "Result 2" section: both provenances, the per-case table, the two independently quotable facts, and the rerun prerequisite (**line numbers move with the version**: 2.8.1 → `:346`, 0.2.3.post2 → `:415`).

**The one substantive revision**
- §1.1 "two generations, two failure modes: SM120 fails loudly, SM121 silently returns all-NaN" → now **split by library version**: on `0.2.3.post2`, SM121 hits **`Assertion error (/deepgemm/csrc/apis/gemm.hpp:415)`** (exit=1) — **silent NaN was not reproduced**. The silent behaviour was recorded on **2.8.1**, and neither 101 nor 102 has a 2.8.1 aarch64 build today ⇒ that claim is **clamped to version granularity**; "SM121 is always silent" can no longer be said.
- §1.1 gains a "SM121 rerun (2026-11-20)" row; the "why standalone" note is tightened accordingly, while the **acceptance conclusion is unchanged**: the failure mode swings between "silent" and "crash" across versions, so "it serves and emits tokens" is still not a passing criterion.
- Side fact: **7 of 8 cases are byte-identical to SM120**, including determinism hash **`f92674f9804a1e90`** ⇒ bf16 masked GEMM is **bit-identical** on SM120/SM121; the divergence is only the FP8 scale-type check.
- §8 gains row 15 ("SM121 silence is an architecture property" → version-scoped); audit table 9 → **10 items**; `docs/N1_v3_executive_summary.en.md` now says "Ten claims".
- Data index row 2 **⛔ → ○** (the rerun artifact ships, but with a different library version than the original record — see the "conditions" column). Distribution: ✔15 / ○21 / ⛔1.

**Still open**
- The original "SM121 silently all-NaN" artifact still does not exist; reproducing it requires building **DeepGEMM 2.8.1** on GB10 (not done as of this release).
- Board items awaiting a decision (`code_output`, SE-03, P3) are out of scope for this release.

## v1.0.6 — 2026-11-20 (L1 artifacts shipped, §6 denominator footnote, two-part release)

**Original artifacts recovered (index row 1 ⛔ → ✔)**
- `scripts/s104_semantic.py` — the actual harness behind N1 §1.1 L1 (10,443 B, 7 cases + 1 subprocess-isolated case), pulled back from 116 `~/dgtest/` on 2026-11-20, unmodified, no absolute paths and no secrets. sha256 `36d3de6a81673fc1b2502f8ca0f333d572ae2dc3fc4d5c7c7964e2ee2484930e`.
- `data/reference_l1_semantic/s104_result_sm120.json` — the 8 `verdict` records from the same box and run (7 `ok`/`suspicious` plus 1 `error` = the `gemm.hpp:346` crash), sha256 `606279055f163886cfb66bdeb0a39b498aeb96c1173543c4532ada608fd5377d`; `README.txt` in that folder gives provenance, the 8-row verdict table and the prerequisite for re-running (DeepGEMM 2.8.1).
- **The SM121 side (index row 2) stays ⛔**: the same script's output on GB10 (`exit=0`, all-NaN, no warning) was never persisted. A full `find *s104*` on 101/102 turns up only `s104_deep_gemm.json` (that is a `deepseek-v4-flash` logprob sampling run, not this harness). The advisory's claim comes from the 2026-11-13 field record; the artifact still has to be re-run.

**§6 / §0.5 denominator made explicit (numbers unchanged)**
- The 6.31 % is relative to **`marlin`** (per pair `(marlin−cutlass)/marlin` = 5.78 % / 7.02 % / 6.13 %); with `cutlass` as the denominator the same data reads **+6.74 %**. The two must not be subtracted or mixed.
- Side finding: **there is no `r1`**; the three pairs are the `sm121` baseline plus `r2` and `r3`. §6 now lists all six artifact filenames (including the per-item jsonl).
- The accuracy side (b=25 / c=20 / p=0.5515, 83.09 % vs 82.71 %) was recomputed item by item from the six `gsm8k_*.jsonl` files and matches the text; the single-pair 44/31, 26/37, 38/31 numbers belong to a different convention, and §8 now says so to stop them being read as "the text was wrong".

**Release shape: one place → two places**
- One release, two assets: `dgx-spark-moe-repro-docs-v1.0.6.zip` (docs / scripts / env / README / CHANGELOG / index / LICENSE) and `dgx-spark-moe-repro-data-v1.0.6.zip` (`data/` — all raw JSON and fingerprints).
- Both zips share the same root folder `dgx-spark-moe-repro/`, so extracting into the same parent merges them; each root carries a `MANIFEST.md` naming the other half and its sha256.

**Where it changed**: `docs/N1_公告_v3_公开版.md` (§6 heading and the "actual pair files" block, throughput denominator, §0.5 parenthetical, §1.1 evidence cell, two new §8 rows, revision-7 table now 9 rows), `docs/N1_v3_executive_summary.en.md` (audit table 7 → 9 rows), rows 1–2 of `N1_data_index.md` / `.en.md`, `README.md` / `README.en.md` (inventory table + two-part distribution note), new `scripts/s104_semantic.py` and `data/reference_l1_semantic/`.

## v1.0.5 — 2026-11-20 (renamed to `dgx-spark-moe-repro` + three-second README)

- **Repository / kit name**: `sm12x-repro` → **`dgx-spark-moe-repro`**. Keeps the `repro` signal while `dgx-spark` / `gb10` carry the search intent; a narrow name is fine because topics cover the rest.
- **Repository description (what GitHub shows in search results)**: `Precision and reproducibility baselines for MoE inference on SM120 (RTX PRO 5000/4500/5080) and SM121 (GB10/DGX Spark), covering vLLM, SGLang, and llama.cpp.`
- **Topics (12)**: `dgx-spark` `gb10` `sm120` `sm121` `nvfp4` `fp8` `moe` `vllm` `sglang` `llama-cpp` `reproducibility` `precision`
- **README opening rewritten**: the positioning prose is gone; it is now a "what is in this repository" table plus a one-line navigation (advisory / closure table / data index / harness / pitfall notes / raw JSON paths / reference runs), Chinese and English in sync.
- **Packaging**: the zip root folder is now `dgx-spark-moe-repro/`; the index status findings from v1.0.4 carry over unchanged.

## v1.0.4 — 2026-11-20 (pre-publication numeric audit, 7 items + a root data index)

**Added**: root `N1_data_index.md` (+ English twin) — 37 rows, one per conclusion, with only "conclusion → raw file → conditions → status". No explanation, no recommendation. Four status marks: ✔ recomputed from the raw file on 2026-11-20 / ○ file present, not recomputed this round / ⏳ on 101/102 (both powered off) / ⛔ physically unreachable or no image.

**Audit (7 items; every number recomputed from the raw JSON, no measured data changed)**

| # | Item | Recomputed result |
|---|---|---|
| 1 | Was the `--enforce-eager` Δlp "rise" decisive? | The 1.899 / 1.905 pairing mixed machines, backends and round counts and is void; same stack 3 vs 3 rounds = **1.258 → 1.905 (+0.647)**, natural SD at that run length **0.071** (120 three-round subsets: mean 1.303, range 1.034–1.529) ⇒ ≈9.1 SD, direction decidable |
| 2 | What "consistent" means in §5.3 | 4 stacks × 10 rounds recomputed per prompt: **13/20 prompts byte-identical in text** and matching BF16, while per-stack logprob identity stays **0/20** ⇒ text only; "80 samples" corrected to **40 runs per prompt** |
| 3 | Source of `p_d ≈ 4.7 %` | same-stack pairings: 56/1319 (4.25 %), **62/1319 (4.70 %, cited)**, 63/1319 (4.78 %), 65/1319 (4.93 %); `SD(b−c) = 7.87` ⇒ threshold 1.17 pp, power 1.67 pp |
| 4 | `num_unit` 10/10 vs 0/20 | 115 `marlin` text always `3584` (10/10) with that stack's logprob identity **0/20** ⇒ "answer right, bits not"; `num_unit` is **not** one of the 13 stable prompts |
| 5 | Absolute level vs relative claim | §5.5 now states the claim is **relative only**; a public SM121 NVFP4 report of ≈86.7 % ± 0.9 % is noted as not re-checked here, and 83.4–83.7 % is **not** claimed as a ceiling |
| 6 | GB / GiB | `71,926,861,539 B ÷ 2^30 = 66.98 GiB`, `48,935 MiB ÷ 1024 = 47.79 GiB`; 65.4 GiB comes from misapplying the 1.1 factor |
| 7 | "TP is not the source" | tightened to **"TP is not an *independent* source"**; the TP × NVFP4-kernel interaction is unseparated, so "TP is irrelevant" is barred (§5.3, §10.2, closure table #2, §8) |

**Where**: `docs/N1_公告_v3_公开版.md` (§5.2 row D, §5.3 wording + per-stack flip counts, §5.4, §5.5 b/c table, §8 three new rows, §10.2 conversion footnote, §12.2, new "revision 7" record), `docs/N1_闭合状态表.md` row #2, the audit table in `N1_v3_executive_summary.en.md`, and the README layout + mapping table.

**Index status recheck (same day, after 101/102 came back up)**: the nine ⏳ rows in `N1_data_index.md` were resolved to concrete files — six now point at files confirmed on disk and moved to ○ (SM121 matrix logs and `s104x3_det_*_r1..10`, `se03_tf2_*_101.json`, the full GGUF Q4/Q8 jsonl, SGLang r2/r3, the BF16-side jsonl); the two L1 rows (`s104_semantic.py` and its SM121-side log) are absent on both 101 and 116 and moved to ⛔ **document excerpt only**; SGLang r1 was not found on disk and that row says so. The §6 accuracy pairing (b=25 / c=20) was still not recomputed and stays ○.

## v1.0.3 — 2026-11-20 (English edition completed; the kit is now bilingual)

**Added, nine English files** (`*.en.md`, sitting next to their Chinese originals, numbers aligned digit for digit, Chinese authoritative):

- `README.en.md`, `CHANGELOG.en.md`
- `docs/NOISE_POLICY.en.md`, `docs/EVIDENCE_INDEX.en.md`, `docs/N1_closure_status.en.md`
- `docs/A8_GSM8K_full1319_conclusion.en.md`, `docs/A8_GGUF_offload_matrix.en.md`, `docs/SE-03_teacher_forcing_verdict.en.md`
- `docs/N1_v3_executive_summary.en.md` — the N1 v3 body (~590 Chinese lines) is covered by an **English executive summary only**; its header states "the Chinese text is authoritative, this summary changes no number"

**Cross-links**: `README.md` gains `English: README.en.md` at the top; `README.en.md` gains `中文: README.md` plus the "Chinese is authoritative" note; `README.md` gains a 9-row Chinese↔English mapping table (including the N1 body → summary exception); the `docs/` listing in `README.en.md` gains the `*.en.md` entries and the two evidence directories `data/reference_replay_1120/` and `data/reference_crossmachine_nvfp4/`.

**Unchanged**: no conclusion, no number, no redaction rule was altered. The Chinese document bodies were not edited (only the README gained the language links and the mapping table).

## v1.0.2 — 2026-11-20 (N1 closure status made explicit; prepared for public release)

**Added**: `docs/N1_闭合状态表.md` — each of the advisory's ten §10 items is now labelled **closed / half-open / open / hardware-unreachable**, and every row states three things: (1) what may be quoted externally today, (2) what is still missing before it can be called closed, (3) where the evidence lives. Tally: **6 closed / 2 half-open / 2 open** (half-open = #3 SM121 still lacks a full-1319 repeat-sampling baseline, #8 GGUF still lacks the non-`-ngl 99` offload levels; open = #4 `cutile`/`b12x` have no image, #5 the GB10 `humming` NVML attribution is unproven).

**Where it is flagged**: the "public-release note" at the top of `docs/N1_公告_v3_公开版.md` now carries a hard rule (judge citability by the status table; do not treat closed items as pending, and do not treat blank items as verified). The `docs/` listing in the README was completed to all seven documents.

**Restated stance**: `N1 v3 + the two 2026-11-20 corrections` = **body closed, three blanks on record**; the advisory stays at **warning level**. It must not be published as "N1 fully closed".

## v1.0.1 — 2026-11-20 (cross-machine rerun: only the NVFP4 cell was reachable, and it proved to be non-deterministic)

**What was done**: a second machine (1× PRO 4500 32 GB / vLLM 0.29.0 / NVFP4 weights only) ran NVFP4 + `marlin` for three rounds using the same `run_matrix.sh`, the same 20 prompts and the same parser as the first machine (2×PRO 5000 48 G, TP=1 on GPU0).

**Results (two sentences, do not merge them)**

1. **Cross-machine first round is byte-identical**: both r1 runs produce content-projection md5 `74dc12605c34eb0206372f5dcd48f02d` (only latency differs) ⇒ changing GPU model or memory size does not affect first-round answers.
2. **Within one server, rounds are not equivalent**: 2 prompts change text on machine A and 3 on machine B, and B's r3 flips `code_bug` from wrong to right ⇒ **NVFP4 / marlin cannot serve as a determinism reference**, corroborating `docs/NOISE_POLICY.md` §2.1 on a second GPU model.

**Left unsolved**: cross-machine evidence for the C3 reference cell (FP8 + `triton` + TP=2) is still missing — the second machine is single-GPU 32 GB without FP8 weights, physically unreachable. README §7.4 records this as a hard gap and forbids passing off "one machine after another" as "cross-machine reproduction".

**New evidence**: `data/reference_crossmachine_nvfp4/` (r1–r3 from both machines + meta + notes).

## v1.0.0 — 2026-11-20 (formal release: Apache-2.0, fully public, replay-verified)

**License**: moved from "internal placeholder" to the **full Apache-2.0** text (including the patent grant), `LICENSE` carries `Copyright 2026 James`; README §5 and this file were rewritten accordingly.

**New evidence documents**: `docs/SE-03_teacher-forcing判定.md`, `docs/A8_GSM8K全量1319_结论.md`, `docs/A8_GGUF卸载门栓矩阵.md` (with the 2026-11-20 revision: the 12-cell subset shows "total constant, per-item not constant"); the "public-release note" at the top of `docs/N1_公告_v3_公开版.md` states that A6/A8 are closed while **not altering any original number**.

**De-identification**: all 48 path occurrences (`~/sm12x`, `/home/james/sm12x`) on the release surface were replaced with `$BASE`; `scripts/attrib_tp_vs_dtype.py` now reads the `BASE` environment variable instead of hardcoding an absolute path. Rescan: IP / hostname / email / phone / MAC / UUID / token all **0**.

**Acceptance (2026-11-20 08:08)**: the archive was transferred, `unzip`-ed into an empty directory (original experiment directory untouched) and `scripts/run_matrix.sh` was run directly on FP8 + `triton` + TP=2 ⇒ `HEALTH_OK` in 80 s, **18/20 × 3**, failures `num_mul` + `logic_weekday`, three-round fingerprint `c8db9a77c6d7b58a40df9d66f75344e7` **byte-identical** to the 11-19 baseline, KV 1,355,776 / 165.50×. Artifacts in `data/reference_replay_1120/`.

**Boundary (important)**: the run above is a **same-machine** (original FP8 box) replay from a freshly unpacked archive, **not cross-machine**. The mechanical side (clone / compile / no leaks) was verified on a **second machine**; cross-machine GPU behaviour (the other SM120 box is offline, SM121 has no FP8 weights) remains undone. README §7.3 records this as a hard constraint and forbids describing 7.1 / 7.2 as "cross-machine reproduction".

## v0.2 — 2026-11-19 10:12 (first real-run acceptance; two claims overturned by measurement)

**Acceptance**: after copying the kit to a **fresh directory** on 115, `scripts/run_matrix.sh` on FP8 + `triton` + TP=2 ⇒ `HEALTH_OK` in 60 s, **18/20 × 3**, three-round fingerprint `c8db9a77…`. Artifacts collected into `data/reference_fp8_tp2/`.

**Two claims overturned by measurement**

1. **`run_matrix.sh` shipped without a tool parser, so the "first run" scored 15/20.** Qwen3.6 emits Qwen3-dialect XML; without `--enable-auto-tool-choice --tool-call-parser qwen3_xml` the API-level `tool_calls` stays empty and the `tc_*` prompts score 0/3 ⇒ 15/20; adding those two parameters ⇒ 18/20 with the other 17 prompts scoring exactly the same. **Both parameters are now script defaults**, and this comparison is recorded in `data/fingerprints/expected.md5 §1.5.1` ("the 3-prompt gap is all parser, not precision").
2. **The fingerprint method missed a field: `tool_calls[*].id`.** The original method hashed the whole `tool_calls` block, but the server mints a new id every run (`chatcmpl-tool-9fac2e5d…`) ⇒ tool-calling runs gave three different digests across three rounds (`76b02bd2…` / `578572a3…` / `5f15b543…`), easily misread as "non-deterministic". **Changed to hash only `type/name/arguments`, sorted** ⇒ one digest for all three rounds, `c8db9a77…`.

**Other updates**: README gained a fourth "read this first" item and one "things not to do" row; a new §7 acceptance record (with the boundary: same machine, different directory, not a fresh machine).

## v0.1 — 2026-11-19 (skeleton, never published)

**In place**

- `env/`: exact versions of the three measured stacks (115 venv 0.29.0 / 116 container 0.31.0+dev / 101–102 0.27.1), image ids, the three refusal types, sampling parameter fields.
- `data/prompts/`: 20-prompt fixed set (3 tool-calling `tc_*`), sha256 registered; grading method and three pitfalls (parser, max_tokens budget, `num_mul` being an inherent error).
- `data/fingerprints/`: **content-projection md5** method plus baseline fingerprints (BF16@TP2 ten rounds `3bfdab3a…`, GSM8K BF16 `c455a539…`, GSM8K FP8 `ab097a76…`) and two **counter-example** fingerprints (the cross-version truth one and the 32/40 one).
- `data/gsm8k/`: fetch-and-slice script, md5, evaluation method (the counting field is `verdict`).
- `scripts/`: `serve_fp8_tp2.sh`, `serve_nvfp4_tp1.sh`, `run_matrix.sh` (serve → three rounds → persist → shutdown, and force-recording of the `effective backend`), plus four scripts taken verbatim from the experiment machine (`probe_20_prompts.py`, `eval_gsm8k.py`, `allreduce_lat.py`, `attrib_tp_vs_dtype.py`).
- `docs/`: `EVIDENCE_INDEX.md` (advisory item → command → raw artifact), `NOISE_POLICY.md` (three evidence classes with three credibility levels + five hard rules).

**Deliberately excluded**

- the 156 GB-class 671B surface (does not fit in 96 G, and putting it in host memory was already rejected);
- SGLang paired throughput (needs a second executor);
- the whole GGUF / llama.cpp family;
- `a6_se03_teacher_forcing.py` (unvalidated, kept out of `scripts/`, only flagged as "not executed" in advisory §10).

**Local limitation (2026-11-19)**: the Windows box had no `git` executable, so the repository was **not initialised** (the tree was ready; `git init` works anywhere git exists, and `.gitignore` already excludes result directories, weights and logs).

**Known gaps**

- No one-shot script for the concurrency curve, only the command text (see `docs/EVIDENCE_INDEX.md §3`).
- De-identification: paths inside the kit are written as `$WORK` / `$BASE` / `$MODEL_DIR`; the `$BASE` definition in `docs/EVIDENCE_INDEX.md` is a placeholder statement (no machine name, no user name). Rescan: all 48 path occurrences on the release surface cleared; IP / hostname / email / phone / MAC / UUID / token **all zero** (2026-11-20).
- `LICENSE` set to **Apache-2.0** (2026-11-20, confirmed by the author, fully public); `docs/N1_公告_v3_公开版.md` is published with the kit as the 2026-11-19 snapshot.

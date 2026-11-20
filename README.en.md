# dgx-spark-moe-repro

**中文**: [`README.md`](README.md) (authoritative) | **English**: this file

> Precision and reproducibility baselines for MoE inference on **SM120 (RTX PRO 5000 / 4500 / 5080)** and **SM121 (GB10 / DGX Spark)**, covering **vLLM, SGLang and llama.cpp**.

## What is in this repository

| Item | Where |
|---|---|
| **N1 precision-safety advisory (body)** | `docs/N1_公告_v3_公开版.md` (Chinese, ~590 lines); English executive summary `docs/N1_v3_executive_summary.en.md` |
| **Closure status table** | `docs/N1_闭合状态表.md` — how far each conclusion can currently be stated, and which cell is still empty |
| **Data index (start here)** | `N1_data_index.md` — each conclusion → raw file / conditions / status, 37 rows, no prose |
| **Reproduction harness** | `scripts/serve_fp8_tp2.sh`, `serve_nvfp4_tp1.sh`, `probe_20_prompts.py`, `eval_gsm8k.py`, `attrib_tp_vs_dtype.py`, `run_matrix.sh` |
| **Pitfall notes / noise discipline** | `docs/NOISE_POLICY.md`, `docs/EVIDENCE_INDEX.en.md`, `docs/SE-03_teacher_forcing_verdict.en.md` |
| **Raw JSON locations** | 115 `$BASE/d4/results/s104i_*`, `s104j_*`, `s104k_*`; 101 `$BASE/d4/results/s104x3_det_*_sm121_101_r{1..10}`; row by row in `N1_data_index.md` |
| **Reference runs for self-checks** | `data/reference_fp8_tp2/`, `data/reference_replay_1120/`, `data/reference_crossmachine_nvfp4/` (meta + expected md5) |

**Three-second navigation**: re-check a number → open `N1_data_index.md`, take its row and the raw file on that machine; run it yourself → §2 quick start; check the claim's boundary → `docs/N1_闭合状态表.md`.

---

## 0. Read this first (otherwise you will waste a run)

1. **Precision results are reproducible; throughput results are not comparable across windows.** On the same stack with the same parameters, the content fingerprint of 20 prompts × 10 rounds should be byte-identical, while tok/s can drift ~18 % within a 3-minute window on the same machine. ⇒ Compare throughput only as **paired ratios inside one window**, never as absolute values. See `docs/NOISE_POLICY.md`.
2. **Always name the MoE backend explicitly** (`--moe-backend marlin|humming|b12x|triton|...`); never rely on `auto`. What `auto` picks varies by version, and unless `effective_backend` is recorded the result cannot be attributed.
3. **Always name the tool-call parser explicitly.** Qwen3-family models emit Qwen3-dialect XML, so only `--tool-call-parser qwen3_xml` populates the API-level `tool_calls` field. With `hermes` parsing fails and silently falls back to plain text (`tool_calls=null`); the grader flags this as `parsed_from_content=True`. **On the same stack, these two parameters alone move the 20-prompt score from 15/20 to 18/20** (the 3-prompt gap is entirely `tc_*`, not precision).
4. **Strip `tool_calls[*].id` before hashing.** The server mints a fresh `chatcmpl-tool-xxxx` per run, so runs that include tool calls look "permanently unstable" if ids are hashed. Measured: three rounds give three different digests with ids, and the same `c8db9a77c6d7b58a40df9d66f75344e7` without them. Method in `data/fingerprints/expected.md5`.

---

## 1. Layout

```
N1_data_index.md       each advisory conclusion → raw file / conditions / status (minimal index; start here)
LICENSE                Apache-2.0
CHANGELOG.md           version history

env/                 version and image inventory (align this layer before reproducing)
  versions.md        exact versions of the three measured stacks: driver / torch / vllm / transformers / flashinfer / NCCL
  requirements.txt   minimal pip-side dependencies (reference only; vllm version follows versions.md)
  docker.md          container images and digests (version-exclusive backends can only be measured there)

data/
  prompts/s104b_prompts.json   20-prompt fixed set (3 of them are tool-calling tc_*), sha256 in fingerprints/
  prompts/README.md            what each prompt tests, grading type, max_tokens
  fingerprints/expected.md5    expected fingerprints (content-projection md5 + raw file md5) with method
  reference_fp8_tp2/           reference run of the same stack (r1–r3 + meta), allows offline fingerprint recomputation
  reference_replay_1120/       same-stack replay from a freshly unpacked copy (r1–r3 + meta + README.txt)
  reference_crossmachine_nvfp4/ NVFP4 + marlin + TP1 on two GPU models (115 / 116, r1–r3 each + meta_*.txt)
  gsm8k/README.md              where the GSM8K 1319 subset comes from, how it is cut, license notes

scripts/
  serve_fp8_tp2.sh             FP8 + triton + TP=2 serve template
  serve_nvfp4_tp1.sh           NVFP4 + marlin + TP=1 serve template
  probe_20_prompts.py          20-prompt grader (with XML-fallback flag)
  eval_gsm8k.py                GSM8K 1319 evaluator (the counting field is verdict, not a boolean correct)
  allreduce_lat.py             two-GPU all_reduce / all_gather latency (must be tee'd to disk to count as measured)
  attrib_tp_vs_dtype.py        TP1 vs TP2 content comparison over ten rounds; emits flip set and Δlp
  run_matrix.sh                one-shot cell: serve → probe → persist → shut down

docs/
  EVIDENCE_INDEX.md            each advisory conclusion → which command / which artifact in this kit
  NOISE_POLICY.md              noise and citation discipline (condensed from advisory §12)
  N1_公告_v3_公开版.md          N1 advisory v3 snapshot (paths rewritten to `$BASE`, with the 2026-11-20 note)
  N1_闭合状态表.md              ★ current status of the ten §10 items: 6 closed / 2 half-open / 2 open
  A8_GSM8K全量1319_结论.md      conclusions and recomputation for the GGUF dual-dtype full 1319 run
  A8_GGUF卸载门栓矩阵.md        12-cell subset matrix (with the 2026-11-20 revision: total constant, per-item not)
  SE-03_teacher-forcing判定.md  SE-03 grey-zone attribution (mechanism = spec-decode)
  *.en.md                      English full translations of the six documents above (N1 body → executive summary)

results/                         empty; store outputs under run_<tag>/, never overwrite someone else's
```

English translations sit next to the originals as `*.en.md`: `EVIDENCE_INDEX.en.md`, `NOISE_POLICY.en.md`, `N1_closure_status.en.md`, `A8_GSM8K_full1319_conclusion.en.md`, `A8_GGUF_offload_matrix.en.md`, `SE-03_teacher_forcing_verdict.en.md`, plus the two root files `N1_data_index.en.md` and `CHANGELOG.en.md`, plus `N1_v3_executive_summary.en.md` — the advisory body itself exists in Chinese only (~590 lines) and that file is its English executive summary. Numbers are aligned digit for digit; where wording differs, the Chinese text wins.

---

## 2. Shortest path (one FP8 TP=2 probe on a dual-GPU box)

```bash
# 0) prepare: model locally, change paths yourself
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-FP8
export WORK=/path/to/this/repo

# 1) serve (background, log to disk)
bash scripts/serve_fp8_tp2.sh "$MODEL_DIR" 8230 results/run_fp8_tp2/serve.log

# 2) wait for health, then run three rounds of 20 prompts
for r in 1 2 3; do
  python3 scripts/probe_20_prompts.py \
    --base-url http://127.0.0.1:8230 --model qwen36-fp8 \
    --tag "fp8_tp2_r$r" --out results/run_fp8_tp2/r$r.json --repeat 1
done

# 3) compare against the expected fingerprint (method in data/fingerprints/expected.md5)
python3 - <<'PY'
import json, hashlib
def canon(p):
    d = json.load(open(p, encoding="utf-8"))
    proj = []
    for r in d.get("results", []):
        tcs = []
        for tc in (r.get("tool_calls") or []):
            f = tc.get("function") or {}
            tcs.append([tc.get("type"), f.get("name"), f.get("arguments")])   # no tc.id
        tcs.sort(key=lambda x: json.dumps(x, ensure_ascii=False))
        proj.append([r["id"], r.get("content") or "", tcs])
    return hashlib.md5(json.dumps(proj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
print(canon("results/run_fp8_tp2/r1.json"))   # FP8@TP2 triton three-round baseline = c8db9a77c6d7b58a40df9d66f75344e7
PY
```

Expected: the **content-projection md5 is identical across all three rounds** (`c8db9a77…`), and the score is **18 / 20** (0.90) with failures fixed at `num_mul` + `logic_weekday` — both of which are also wrong in BF16, so this is **not** a quantization loss. The kit ships the same-stack reference run in `data/reference_fp8_tp2/` (r1–r3 + meta) so the fingerprint can be recomputed offline.

---

## 3. Standard procedure for "is this backend deterministic?"

```bash
# A. ten rounds on one stack (same parameters, only the round index changes)
for r in $(seq 1 10); do
  python3 scripts/probe_20_prompts.py --base-url ... --model ... \
    --tag "det_r$r" --out results/run_det/r$r.json --repeat 1
done
python3 scripts/attrib_tp_vs_dtype.py --glob "results/run_det/r*.json"

# B. move one axis (TP only, dtype unchanged), ten more rounds → is dtype or TP the source?
for r in $(seq 1 10); do
  python3 scripts/probe_20_prompts.py --base-url ... --model ... \
    --tag "tp2_r$r" --out results/run_tp2/r$r.json --repeat 1
done

# C. BF16 control (same version, same TP)
#    note: BF16 weights are 71.9 GB and do not fit on one 48 G card → TP=1 is hardware-unreachable, not "not done"
```

Criteria:
- content-projection md5 identical across ten rounds ⇒ that stack is deterministic (usable as a reference);
- any flip ⇒ report the **flip set + max Δlp + number of distinct trajectories**, never just "sometimes right, sometimes wrong".

---

## 4. Things not to do (learned the hard way)

| Don't | Why |
|---|---|
| Subtract tok/s from two different windows | ~18 % drift in 3 minutes on the same box; the difference is not an effect |
| Compute a speedup from two independent regression slopes | Only paired ratios (same window, alternating) + win count + permutation p are allowed |
| Put BF16 weights in host memory and call it "runs" | Comms and bandwidth all change, so the measurement loses meaning |
| Treat `is_supported()` as a list of usable backends | It answers "can it compile", not "does it satisfy the QuantKey contract"; `cutlass` and `emulation` were both probed and both refused at serve time |
| Use `pgrep -f 'vllm serve'` to decide whether a server is running | It matches your own `ssh bash -c` command line, a guaranteed false positive |
| Use `±5 pp` as a pass gate | Single-run standard error on 1319 items is ≈1.0 pp, so a 5 pp gate filters nothing |
| Hash the whole `tool_calls` block including server-generated ids | Ids change every run, turning "deterministic" into "non-deterministic"; strip ids, keep only `type/name/arguments` |

---

## 5. License and data

- Code: **Apache-2.0**, full text in `LICENSE` (includes the patent grant; commercial use and redistribution allowed, copyright and disclaimer must be retained).
- `s104b_prompts.json` is a self-authored prompt set and may be redistributed with the kit; `gsm8k_test_1319.jsonl` comes from GSM8K (MIT), so the kit ships only the **fetching script and fingerprints**, not a full copy of the data.
- All paths inside the kit are written as `$WORK` / `$MODEL_DIR`; **no machine names, internal IPs, or absolute home directories**.

## 6. Known gaps (honest list)

- The concurrency curve (c=1…24) exists only as measured numbers and logs, **there is no one-shot script** (the exact `vllm bench serve` invocations are quoted in `docs/EVIDENCE_INDEX.md`).
- The GGUF / llama.cpp family is **not covered by this kit at all**.
- The SM121 (GB10) reproduction scripts are shared with SM120, but the **image version differs**; see `env/docker.md`.
- The `a6` teacher-forcing (grey-zone attribution) script was never validated and is **not included in `scripts/`**.
- **Cross-machine reproduction of the FP8 + `triton` + TP=2 cell cannot be done**: the only other SM120 machine is single-GPU 32 GB without FP8 weights, and the SM121/GB10 machines also lack FP8 weights. Only the NVFP4 cell can be extended across machines (see §7.4, and it was proven non-deterministic).

## 7. Acceptance record

### 7.1 Same machine, fresh directory (2026-11-19)

The kit was copied wholesale into a **fresh directory** (not the original experiment directory) and then run with `scripts/run_matrix.sh` on FP8 + `triton` + TP=2:

| Criterion | Result |
|---|---|
| Serve | `RESULT=HEALTH_OK`, healthy in 60 s; KV 1,355,776 tokens / 8,192 ctx ⇒ 165.50× |
| Effective backends | `Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod` + `Using TRITON Fp8 MoE backend` |
| Grading | **18 / 20 × 3**, failures `num_mul` + `logic_weekday` (both also wrong in BF16) |
| Fingerprint | identical across three rounds: `c8db9a77c6d7b58a40df9d66f75344e7` |
| Path portability | only `MODEL_DIR` and the port changed; no path reached back into the original directory (`$BASE`) |

**Honest boundary**: this is "same machine, different directory", demonstrating **path portability and absence of hidden dependencies**; it is **not** a "fresh machine" (the same venv and models were still installed).

### 7.2 Replaying from the release archive (2026-11-20)

`sm12x-repro-v1.0.0.zip` (sha256 `7942b38788d39d4185b43b186566fb6da79892f7d6d2871d88f6aff045fe30c6`, 29 files) was transferred, `unzip`-ed into an empty directory, **without touching the original experiment directory `$BASE`**, and `scripts/run_matrix.sh` was run directly on the same cell (FP8 + `triton` + TP=2):

| Criterion | Result |
|---|---|
| Archive integrity | sha256 matches the published value; 29 files; `py_compile` clean |
| Serve | `RESULT=HEALTH_OK`, healthy in 80 s |
| Effective backends | `Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod` + `Using TRITON Fp8 MoE backend` |
| KV | 1,355,776 tokens / 8,192 ctx ⇒ 165.50× |
| Grading | **18 / 20 × 3**, failures `num_mul` + `logic_weekday` (exactly the same set as 7.1) |
| Fingerprint | identical across three rounds: `c8db9a77c6d7b58a40df9d66f75344e7`, **byte-identical** to the 7.1 baseline |
| Evidence | `data/reference_replay_1120/` (r1–r3 + `meta.txt`) |

### 7.3 What these two runs actually prove (must read, do not extrapolate)

- 7.1 and 7.2 ran on the **same 48 GB machine** (the original FP8 experiment box); only the directory and the unpacked copy changed ⇒ they prove that the kit **unpacks from an archive into a clean tree, is self-consistent, and reproduces the baseline byte-exactly**, and they are **not** "cross-machine reproduction".
- The **mechanical** side was verified on a **second machine**: `git clone` → 29 files → `python3 -m py_compile scripts/*.py` clean → path-leak check `LEAK_FREE` → `probe_20_prompts.py --help` works.
- **Cross-machine GPU behaviour of the FP8 cell is still unverified**: the other SM120 machine is **single-GPU 32 GB without FP8 weights** and physically cannot run 2-way FP8 — see the substitute experiment and its conclusion in 7.4. Until a cross-machine run is done per `env/versions.md`, **do not describe 7.1 / 7.2 as "cross-machine reproduction"**.

### 7.4 Cross-machine rerun (2026-11-20, second GPU model): only the NVFP4 cell is reachable, and it is not deterministic

The second machine is 1× RTX PRO 4500 32 GB with vLLM 0.29.0 and only NVFP4 weights in its model directory ⇒ the FP8 + TP=2 cell is **physically unreachable**. So the same `run_matrix.sh`, the same 20 prompts, the same parser and the same version were used to run NVFP4 + `marlin` + TP=1 for three rounds on both machines:

| Machine | r1 | r2 | r3 | Same md5 across rounds |
|---|---|---|---|---|
| A: 2×PRO 5000 48 G (GPU0 only) | 17/20 | 17/20 | 17/20 | no |
| B: 1×PRO 4500 32 G | 17/20 | 17/20 | **18/20** | no |

Two conclusions that must be stated separately:

1. **Cross-machine first round is byte-identical**: both r1 runs produce content-projection md5 `74dc12605c34eb0206372f5dcd48f02d`, differing only in `latency_s` ⇒ changing GPU model and memory size does not affect the first-round answers.
2. **Within one server, rounds are not equivalent**: A shows 2 prompts with changed text, B shows 3, and B's r3 flips `code_bug` from wrong to right (17/20 → 18/20) ⇒ **the NVFP4 / marlin cell cannot serve as a determinism reference**. This corroborates `docs/NOISE_POLICY.md` §2.1 ("NVFP4 variance is kernel non-determinism") on a second GPU model.

Evidence: `data/reference_crossmachine_nvfp4/` (r1–r3 from both machines + per-machine meta).

**One-line ledger**: the only cell usable as a determinism reference is FP8 + `triton` + TP=2 (always `c8db9a77…` across three rounds), and its cross-machine evidence could not be produced this time; the NVFP4 cell that could be extended happened to prove that this cell is non-deterministic. **This gap stays in the kit as-is; never dress up "one machine after another" as "cross-machine reproduction".**

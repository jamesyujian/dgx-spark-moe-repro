# N1 advisory · closure status (2026-11-20)

*English version of `docs/N1_闭合状态表.md`. The Chinese file remains the reference.*

> This table is the **current status** of §10 "not covered / pending" in `docs/N1_公告_v3_公开版.md` and replaces the old "pending list" reading of that section.
> Rule: **only items marked "closed" may be cited externally as done**; "half-open" items may cite only the completed half, with the missing half stated; "open" items must not be written as "measured/verified"; "unreachable" items must not be written as "not done".
> Version anchor: `N1 v3 (revision 6)` snapshot + the two 2026-11-20 corrections (A6/SE-03, A8 full 1319).

## 1. Status of the ten items

| # | Original item | Status | What may be stated externally | What is still missing to close it | Evidence |
|---|---|---|---|---|---|
| 1 | BF16 GSM8K was run only 3 times (1105 / 1120 / 1120) | **Closed (closed on information content)** | May state "greedy decoding is deterministic on BF16": three runs are **byte-identical**, md5 `c455a539e5a230171220418061f053f5`, so rounds 4/5 carry zero information | No rerun needed. If the phrase "5 rounds" is required, write "3 rounds proved determinism; rounds 4/5 not run" | Three-run md5 + computed 139-item intersection (set 115: 8 items / 0.61 pp; set 116: 5 items / 0.38 pp; the two sets must not be mixed) |
| 2 | Determinism of BF16@TP=1 not measured | **Closed (one cell recorded as hardware-unreachable)** | May state "BF16@TP=1 = hardware-unreachable": weights 71,926,861,539 B = 66.98 GiB > 47.79 GiB per card (48,935 MiB). Substitute attribution completed — NVFP4 TP=1 flips 5/20, TP=2 flips 6/20 (intersection 5 prompts), BF16@TP=2 flips 0/20 ⇒ **TP is not an *independent* source** (**must not** be written as "TP is irrelevant": the TP × NVFP4-kernel interaction was never separated) | The BF16@TP=1 cell **will never have data** on this hardware (does not fit); a larger card would require redoing it | `s104h_nvfp4_tp2_r{1..10}.json`, `s104i_base_marlin_r{1..10}.json`, `s104k_bf16_r{1..10}.json` |
| 3 | SM121 (101/102) lacks a repeat-sampling baseline | **Half-open** | May state "the 20-prompt × 10-round Δlp baseline on SM121 exists" (four stacks, identical logprobs 0/80, Δlp 1.623 / 1.623 / 2.089 / 2.006) ⇒ §5 scope already covers both architectures | **Full-1319 repeat sampling** not done; **GPU-model × version full matrix** not done | `d4r_nvfp4_*_sm121_*.json` |
| 4 | No data for `flashinfer_cutile` at 0.31.0 + ≥48 G; none for `flashinfer_b12x` at 48 G + 0.31.0 | **Open (hard blocker, no image)** | May state only "**no data**". Must not be written as "unusable", nor as "refused" | Needs an image/build that can load that backend; the current environment has none ⇒ no short-term path | Explicit no-image table (advisory §4.1) |
| 5 | `humming` crashes on GB10 with an NVML error; whether it can be bypassed is untested | **Open (symptom yes, attribution is inference)** | May state "on GB10 it **crashes at serve time** with `pynvml.NVMLError_NotSupported`, which is neither a device refusal nor a QuantKey refusal". ⚠ **Must not present "the kernel reads NVML fields before selection" as fact** — there is no stack trace and no bypass experiment, it is inference | (1) a full stack trace; (2) a bypass experiment (e.g. an NVML shim or disabling that query) | The `NVMLError_NotSupported` text; advisory §4.1 and §10 item 5 use the same wording |
| 6 | SE-03: `b12x` vs `deep_gemm` first-token difference >3.4 nats on `num_mul` in the mxfp4 stack | **Closed (2026-11-20)** | May state "**the mechanism is spec-decode**: with spec disabled, chat generation matches teacher-forcing". That stack was 4/4 content-identical (deterministic), so it never belonged to the §5 non-determinism conclusion | None. Promoting it to a standalone SE entry would need its own number (not done) | `docs/SE-03_teacher-forcing判定.md` |
| 7 | `flashinfer_cutedsl` W4A16 verified unusable only on SM121 | **Closed (independently reproduced on two machines)** | May state "**SM120 + `flashinfer_cutedsl` is device-level unavailable**": 116 (0.31.0) refuses at 40 s, 115 (0.29.0) at 20 s, with the same `kernel does not support current device cuda`; the refusal happens before weight loading ⇒ not capacity, not QuantKey | None. Claiming "permanently unusable" would need a retest after the upstream device whitelist actually changes | `logs/a7_cutedsl_sm120{,_container}.log` (278 lines) |
| 8 | GGUF: three Q8_0 levels and `-ngl 32` / `-ngl 0` reruns "to fill in" | **Half-open** | May state "`-ngl 99` is fully done": Q4_K_M and Q8_0 both 1319/1319, both **1128/1319 = 0.855193**, zero syntax-invalid; flips **58 prompts (4.40 %) symmetric both ways** (Q4-only 29 / Q8-only 29), McNemar **p = 1.0000** ⇒ the two levels are **indistinguishable**; about 3/4 of flips come from 512-token truncation | **`-ngl 32` / `-ngl 0` full runs not done**; **Q4 same-config rerun (drift baseline) not done**; `max_new_tokens 768 / 1024` untested ⇒ must not claim "GGUF bit-exactness fully verified" | `full_q4ngl99full.jsonl`, `full_q8ngl99full_102.jsonl`, `A8_full_report.md/json`, `docs/A8_GSM8K全量1319_结论.md` |
| 9 | c1 / c2 effective levels not run; "optimal concurrency = 4" | **Closed (2026-11-19)** | May state the full curve: c=1…24, **209 → 1011 tok/s (4.84×) unsaturated**, the ceiling being `--max-num-seqs 32` rather than compute; single-user decode TPOT 4.17 → 17.4 ms (4.2×); **c=6 is a dip** (inferred: it falls on an uncaptured cudagraph size) | None. ⚠ Still bound by the hard rules: never subtract from 11-18 absolute throughput; c≤8 and c≥12 used different load sizes, so no strict cross-level pairing | `d4/logs/a9_concurrency_sweep_v{2,3}.log`, `a9_c*_r*.log`, `a9b_c*_r*.log` |
| 10 | The 20-prompt probe was run without `tools` | **Closed (2026-11-19)** | May state "tool calling works on both SM120 stacks, **provided `--tool-call-parser qwen3_xml` is explicit**": three rounds per stack, `tc_*` **3/3**, all real API-level `tool_calls`, `parsed_from_content = 0`; with `hermes`, `tool_calls` is null (old tc conclusions count at the text layer only) | None | `d4/results/a10e_*`, `d4/logs/a10*` |

## 2. Tally

| Status | Count | Items |
|---|---|---|
| **Closed** | 6 | #1, #2, #6, #7, #9, #10 (#1 / #2 are "closed on information content"; #2 contains one cell recorded as hardware-unreachable) |
| **Half-open** | 2 | #3 (missing full-1319 repeat sampling and the GPU-model × version matrix), #8 (missing `-ngl 32/0`, the Q4 rerun, and the 768/1024 budget) |
| **Open** | 2 | #4 (no image, hard blocker), #5 (the NVML attribution is still inference, unproven) |

## 3. Release-surface gaps outside §10 (C3)

| Gap | Status | How to write it externally |
|---|---|---|
| **Cross-machine** reproduction of the FP8 + `triton` + TP=2 cell | **Cannot be produced** | May state only "same machine, fresh directory (7.1) + replay from the release archive (7.2), fingerprints byte-identical"; **must not** state "cross-machine reproduced". The second SM120 machine is single-GPU 32 GB without FP8 weights; SM121/GB10 has no FP8 weights either |
| Cross-machine recheck of the NVFP4 cell | **Done (2026-11-20)** | May state "the **first round** on both machines has a byte-identical content-projection md5 (`74dc1260…`)", but **rounds within one server are not equivalent** (2–3 prompts change text, one verdict flip) ⇒ this cell **cannot serve as a determinism reference** |
| `code_output` single-GPU 6 / TP2 4 | **Not separated** | Still inside the original table, never promoted to a standalone SE entry |
| Whether SE-03's spec finding becomes a standalone formal entry | **Not split** | Currently folded into `docs/SE-03_teacher-forcing判定.md` |

## 4. One-page conclusion

- **Safe to announce as "closed"**: the three-layer FP8 split and the L2 contract; NVFP4 non-determinism (including TP attribution and the BF16 control); the concurrency curve; the tool-calling wording; the SE-03 mechanism; `cutedsl` being device-level unavailable; GGUF `-ngl 99` dual-dtype full run (including the 58 flips and the truncation attribution).
- **Must stay blank**: cutile / b12x (no image); GB10 `humming` NVML attribution; GGUF levels other than 99 and the Q4 rerun; SM121 full-1319 repeat sampling; cross-machine FP8.
- **One-line stance**: `N1 v3 + two 2026-11-20 corrections`, **body closed, three blanks on record**, the advisory remains at **warning level**.

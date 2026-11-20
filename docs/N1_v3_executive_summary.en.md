# N1 advisory v3 — English executive summary

*This is an English summary of `docs/N1_公告_v3_公开版.md` (Chinese, ~590 lines), which remains the authoritative and complete text. Anything cited externally must resolve to the Chinese document; if the two disagree, the Chinese text wins. Nothing in this summary changes any number.*

- Published: 2026-11-17 | Version **v3 (revision 6)** | Status: **warning level** (the severity of "NVFP4 is not reproducible" was raised one notch relative to v2)
- Snapshot shipped with this kit: the 2026-11-19 v3 text, with paths rewritten to `$BASE` and a 2026-11-20 note recording two later corrections (A6/SE-03 and A8)
- Closure status of all open items: `docs/N1_closure_status.en.md` — **6 closed / 2 half-open / 2 open**, the advisory stays at warning level

## The five headline conclusions (§0)

### 1. FP8 must be split into three layers by "path + who owns the scale"

The difference is **not** about "block-wise or not", but about **who holds the scale and who performs the scale conversion**. The same weight batch crashes when the engine disables the cast, and runs when the framework casts internally — so layering must follow path and scale ownership, not the quantization name.

| Layer | Numeric contract (scale ownership) | Engine / path | SM120 | SM121 |
|---|---|---|---|---|
| **L1** | f32 / non-ue8m0 scale fed **directly** to the engine (`use_ue8m0=False`, caller disables the cast) | DeepGEMM 2.8.1 direct | **hard failure** (`Assertion ... gemm.hpp:346` or `CUDA 719`) | **silently all-NaN** (`exit=0`, no warning) |
| **L2** | **W8A8 block-scaled**: weights with `weight_block_size=[128,128]` and **f32 `weight_scale_inv`** (31,745 of them) + dynamic per-token activation; **cast done inside vLLM** | vLLM 0.29.0: dense via `CutlassFp8BlockScaledMMKernel`, MoE via `--moe-backend triton` / `deep_gemm` | **serves and runs**: TP1 34/40, TP2 **36/40**, GSM8K 1319 = **0.8453** | not measured |
| **L3** | same block-scale weights, **other MoE backends** | `flashinfer_cutlass` / `cutlass` / `hpc` | **refused**, reason **not evidenced** | not measured |

Citation discipline (hard rules):
- saying "FP8 cannot be used" **must** name **the L1 path**; saying "FP8 can be used" **must** name **L2 plus its four boundaries**; the two sentences may not borrow from each other;
- "forbidden" holds only for L1 (DeepGEMM direct + non-ue8m0) — the same weights run after vLLM's internal cast;
- "usable" holds only for L2 and always with the four boundaries;
- L3 is "refused, reason not evidenced": logs show only the **expected** `QuantKey`. It must **not** be written as "layout mismatch" (the measured layout matches `GroupShape(128,128)` exactly), nor as "unusable".

**L2 accuracy detail**: GSM8K 1319 on L2 = **0.8453**; BF16 at the same version and TP = **0.8491**; difference **−0.38 pp**, McNemar 32/37, **p = 0.63** ⇒ indistinguishable. The 20-prompt failure set on TP2 matches the `s104b_bf16.json` reference (TP2, 11-15), i.e. the same prompts fail in BF16.

### 2. `marlin` is the only NVFP4 MoE backend that works on all *measured* machine × version combinations

Measured: PRO 5000 48 G × 0.27.1 / 0.29.0 / 0.31.0 (9 runs each), PRO 4500 32 G × 0.31.0, GB10 × 0.27.1. **Of the 3×3 = 9 cells, only 5 were measured**; the other 4 (32 G × 0.27.1/0.29.0, GB10 × 0.29.0/0.31.0) **have no matching image on the machines and therefore no data** (checked via `docker images` on 2026-11-18). It must **not** be extrapolated to "all 3×3 pass", and every citation of this item **must carry the version number** — no single-version result may be phrased as an architectural conclusion.

### 3. The failure mode that matters for NVFP4 is "same configuration, repeated calls, different answers" — not NaN

Same machine, same version, same process, same prompt, `temperature=0`, `seed=0`: BF16 logprob difference is **exactly 0.000** (3 rounds 20/20, 10 rounds 20/20), while NVFP4 reaches **1.26–2.09 nats** with **0–1/20** identical. Reproduced on **both SM120 and SM121**. The root cause sits **inside the MoE kernel (split-K / atomic reduction)**; the scheduling layer was excluded by two experiments, and **no deployment parameter avoids it** (`VLLM_BATCH_INVARIANT=1` fails at serve time on this checkpoint).

Severity raised relative to v2: v2 described "some configuration is biased" (avoidable by switching backends); the current finding is "**same configuration, same process, `temperature=0`, `seed=0`, repeated calls are not reproducible**" — the mitigation moved from "switch backend" to "must verify batch invariance", and that mitigation is unavailable on this checkpoint.

### 4. `num_unit` on "0.31.0 + `marlin` + 32 G" is about 27.5 % correct (11/40, 95 % CI 16–43 %) and unstable even within one process — not a "process-level bias"

Ten identical repeats inside one round produced 4 distinct answers (`3584`×1, `3686.4`×3, `3626880`×5, `3626.88`×1); there are 4 distinct wrong values, and 8/40 are **~10³ magnitude errors** (`3626880` × 6, `3635978.24` × 2). On the same version and machine, `humming` scores 8/10 (Fisher exact p = **0.0055** against marlin's 1/10), while 115 / 0.29.0 `marlin` scores 10/10 ⇒ **combination-specific**. It must **not** be written as "marlin is worse". The cross-process homogeneity check (4/10, 6/10, 0/10; χ² = 5.60, df = 2, **p = 0.061**) does not support "process-level bias", and on the full 1319 both backends have **0** backend-specific hard errors.

### 5. On SGLang, switching the MoE backend from `marlin` to `flashinfer_cutlass`: accuracy indistinguishable (per-item McNemar b=25 / c=20, exact **p = 0.55**), throughput down 6.31 %

89.46 → 83.81 tok/s, n = 3 vs 3, internal jitter **0.95 % / 0.37 %**, direction consistent across three pairs ⇒ a decidable **performance fact**, not a precision issue. The original claim "no accuracy difference, 0.227 pp" was the difference of two 3-run means (= 3 items), **smaller than the backend's own range 0.83 / 0.99 pp**, and is no longer cited as a measured difference.

## Where the rest lives

| Topic | Chinese section | English version |
|---|---|---|
| FP8 three layers, L2 boundaries, L3 refusals | §1.0–§1.3 | summary above + `docs/N1_公告_v3_公开版.md` |
| NVFP4 is a different family; failure-mode typing | §2–§3 | — |
| Backend availability matrix (vLLM / SGLang, by machine × version; never merged) | §4.1–§4.4 | — |
| Reproducibility: five stacks on SM120, four on SM121; root cause; `num_unit`; GSM8K | §5.1–§5.5 | — |
| Performance: SGLang `marlin` 6.31 % faster than `flashinfer_cutlass` | §6 | — |
| Self-check list (5 executable checks) | §7 | — |
| Retired / clarified claims (do not keep circulating) | §8 | — |
| Verification coverage (who, where, what) | §9 | — |
| Not covered / pending (10 items) | §10 | `docs/N1_closure_status.en.md` |
| Noise and citation discipline | §12 | `docs/NOISE_POLICY.en.md` |

## Corrections folded in (2026-11-20)

1. **A6 / SE-03** is closed: the mechanism is **spec-decode** — with spec disabled, chat generation matches teacher-forcing; with spec enabled, the verification path substitutes the model's own preference. Consequence: on that stack, "backend A answered correctly" was not a more accurate kernel but a by-product of spec. See `docs/SE-03_teacher_forcing_verdict.en.md`.
2. **A8 / GGUF** is closed at `-ngl 99`: full 1319, Q4_K_M = Q8_0 = **1128/1319 = 0.855193**, 58 flips (4.40 %) symmetric, McNemar p = 1.0000 ⇒ indistinguishable; about 3/4 of flips come from 512-token truncation. The earlier "accuracy constant across 12 cells" must now be read as "**total constant, per-item not**". See `docs/A8_GSM8K_full1319_conclusion.en.md`.

## Pre-publication numeric audit (v3 revision 7, 2026-11-20)

Seven claims were recomputed from the raw JSON. No measured data changed; wording, pairing and counting did.

| # | What was audited | Recomputed result | Where it was fixed |
|---|---|---|---|
| 1 | Was the `--enforce-eager` Δlp "rise" decisive? | The published pairing (1.899 / 1.905, Δ = 0.006) mixed machines, backends and round counts. Same stack (115 / 0.29.0 / `humming`, 3 vs 3 rounds): **1.258 → 1.905 = +0.647 nats**. Natural SD of Δlp at that run length is **0.071** (all 120 three-round subsets of the ten rounds: mean 1.303, range 1.034–1.529) ⇒ **≈9.1 SD**, the direction is decidable | §5.2 row D, §12.2, new invalid-pairing row in §8 |
| 2 | What does "consistent" mean in §5.3? | Recomputed per prompt: with 4 stacks × 10 rounds, **13/20 prompts are byte-identical in final text** and match BF16's text, while per-stack logprob identity stays **0/20** ⇒ "consistent" means **final answer text only**. The "80 samples" figure is an arithmetic error: 4 × 10 = **40 runs per prompt** | §5.3 (wording block + count corrected) |
| 3 | Where does `p_d ≈ 4.7 %` come from? | Same-stack run-to-run per-item pairing: `029` r4/r5 = 56/1319 (4.25 %), **`031` r4/r5 = 62/1319 (4.70 %, the cited value)**, cross-group 63/1319 (4.78 %), 65/1319 (4.93 %). `SD(b−c) = 7.87` ⇒ threshold 1.17 pp, 80 % power 1.67 pp | §5.5 (b/c table added) |
| 4 | Does `num_unit` 10/10 contradict 0/20? | On 115 `marlin` the text is always `3584` (10/10) while that stack's logprob identity is **0/20** ⇒ "answer right, bits not"; `num_unit` is **not** among the 13 stable prompts | §5.4 |
| 5 | Absolute level vs relative indistinguishability | A public SM121 NVFP4 report (0-shot / chat template / thinking disabled / flexible extraction) states ≈ 86.7 % ± 0.9 %; **its raw artifacts were not re-checked here** | §5.5 (explicit "relative only" limit; no claim that 83.4–83.7 % is a ceiling) |
| 6 | GB vs GiB | `71,926,861,539 B ÷ 2^30 = 66.98 GiB`; `48,935 MiB ÷ 1024 = 47.79 GiB`; 65.4 GiB only appears if the 1.1 factor is misapplied | §10.2 footnote |
| 7 | "TP is not the source" | What the experiment rules out is "TP as an *independent* source"; the TP × NVFP4-kernel interaction was never separated | §5.3 / §10.2 tightened; §8 wording row |

Row-by-row pointers from every conclusion to its raw file, conditions and status are in the repository root, `N1_data_index.md` / `N1_data_index.en.md`.


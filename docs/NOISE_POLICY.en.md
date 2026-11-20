# Noise & citation discipline (condensed from advisory §12)

*English version of `docs/NOISE_POLICY.md`. The Chinese file remains the reference.*

## 1. Three evidence classes, three credibility levels

| Class | Example | Credibility | Citation discipline |
|---|---|---|---|
| Determinism / precision | ten-round content-projection md5, per-prompt verdicts, Δlp, identical logprobs | **High** | absolute values may be quoted, **the full stack identity must be stated** (GPU model / SM / version / backend / dtype / TP) |
| Paired ratios | backend A vs B at −6.31 %, A/B win count | **Medium** | quote only the **ratio + win count + p**; never quote the two absolute sides; ratios from different windows **must not be subtracted** |
| Absolute throughput | single-run tok/s, throughput at some concurrency level | **Low** | always median + range + window, and mark "not comparable across windows" |

## 2. Three quantified noise sources

1. **The NVFP4 variance is kernel non-determinism, not sampling noise.** With `temperature=0` and a fixed seed, ten rounds still flip 5–7 prompts; BF16 on the same machine is byte-identical across ten rounds (0 flips). Two exclusion experiments: `VLLM_BATCH_INVARIANT=1` **fails at serve time**; with `--enforce-eager` Δlp actually **rises**. **Independently re-checked on a second GPU model (PRO 4500 32 G) on 2026-11-20**: within one server, 2–3 prompts change text across three rounds and `code_bug` flips from wrong to right (17/20 → 18/20); yet the **first round** on both machines has a byte-identical content-projection md5 (`74dc1260…`). ⇒ This cell cannot be used as a determinism reference, and a matching first round does not imply reproducibility.
2. **Absolute throughput drifts**: ~18 % within a 3-minute window on the same machine and stack (first quantified on the 5080 node).
3. **The A/B pairing lacks a true regression positive control**: it can only show "the difference is ≥ the noise floor", never "it can detect a known regression". ⇒ **Resolution floor = tolerance**.

## 3. Five hard rules

1. Never subtract across versions (GSM8K references must be same version, same TP).
2. Never compare ratios across windows.
3. Never claim a quantized path "beats BF16": any Δ below tolerance must be written as "indistinguishable".
4. State the sample size openly: single-run standard error on 1319 items is ≈1.0 pp ⇒ a `±5 pp` pass gate is void.
5. For concurrency/throughput, describe only the **curve shape and knee**, never "absolute performance of some level is good or bad".

## 4. Minimum information for reproduction (miss one and numbers cannot be attributed)

`GPU model + SM` | `engine version (down to the dev string)` | `MoE backend (explicitly named)` | `TP/PP/DP` | `dtype + quantization contract` | `ctx / max-num-seqs / util / enforce-eager / batch-invariant` | `prompt-set version + grading script + md5 per round`

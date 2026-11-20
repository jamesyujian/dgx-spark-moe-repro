# A6 · SE-03 teacher-forcing verdict (deep_gemm / b12x two-sided comparison)

*English version of `docs/SE-03_teacher-forcing判定.md`. The Chinese file remains the reference.*

- Date: 2026-11-19
- Hosts: **101 (GB10 / SM121) head + 102 worker**, TP=2 / `--nnodes 2`, served-model `deepseek-v4-flash`, port 8888
- Stack: `anemll/dspark-vllm-gx10:0.1.1`, DeepSeek-V4-Flash-DSpark, `quantization=deepseek_v4_fp8` (MoE experts mxfp4), `--speculative-config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}`, `--enable-prefix-caching`, `kv-cache-dtype nvfp4_ds_mla`
- Prompt: `计算 1234567 × 89 的结果，只输出数字，不要解释。`; correct value **109876463**; the wrong value recorded for b12x in SE-03 is **109876543**

## 1. Method (settle the method before quoting numbers)

Both prerequisite pitfalls were hit in measurement, and the handling is pinned down in `se03_tf2.py`:

1. **A raw `/v1/completions` prompt cannot take the original text**: this stack treats the text as document continuation and emits `"\n}<｜end▁of▁file｜>`, at which point every `prompt_logprobs` absolute value is void (the first token produces numbers like −16.24, three orders of magnitude off from 11-13).
2. **You must obtain the templated `token_ids` from `/v1/chat/completions/render`** (22 tokens for this prompt: `[0,128803,4339,223,6895,18009,25,2863,223,4362,223,17488,303,1299,10626,8283,303,4916,9068,320,128804,128822]`, the last 128822 being the assistant start), then send that **id list** as `prompt` to `/v1/completions` (`max_tokens=1`, `prompt_logprobs=1`).
3. **`prompt_logprobs[0]` is always null (BOS)**, so alignment must be positional (`prompt_logprobs[i]` ↔ the i-th token); `max_tokens=0` is rejected with 400 ("must be at least 1").
4. The candidate token count comes from `/tokenize`; a candidate's score is the mean logprob over its trailing positions.

**Method self-check**: the same method gives first token `109` = **−0.0032** on b12x (11-13 recorded −0.0054, same order), and `123` = **−1.0280** on deep_gemm (11-13 recorded −1.028, **digit-for-digit match**).

## 2. First token (group A) — the two backends prefer opposite values

| Backend | `109` | `123` | Preference | Margin |
|---|---|---|---|---|
| `flashinfer_b12x` | **−0.0032** | −5.3797 | **109** | 5.3765 nats |
| `deep_gemm` | −3.9030 | **−1.0280** | **123** | 2.8750 nats |

**Conclusion 1 (primary criterion)**: SE-03's divergence sits at the **first token**, and the two backends point in opposite directions with margins of 5.38 / 2.88 nats — **not a near tie, not sampling noise**. The first-token choice determines the rest of the path: picking `109` ⇒ "just give the number"; picking `123` ⇒ "write the expression first".

**Generation agrees with this** (one run each, temp=0):

| Backend | Generated output | First-token branch | Verdict |
|---|---|---|---|
| b12x | `109876543` | 109 (wrong path) | wrong (off by 80) |
| deep_gemm | `1234567 × 89 = 109876463` | 123 (expression path) | correct |

## 3. Full answer and trailing digits (groups B / C) — same direction, but at odds with generation

**B full answer (mean token logprob)**

| Backend | correct `109876463` | wrong `109876543` | per-position lp (correct / wrong) | wrong leads by |
|---|---|---|---|---|
| b12x | −2.0048 | **−0.0067** | `109`: −0.0053 / −0.0048; `876`: −0.0019 / −0.0073; `463/543`: **−6.0071 / −0.0078** | 1.998 nats |
| deep_gemm | −4.0966 | **−1.4299** | `109`: −3.903 / −3.903; `876`: −0.3074 / −0.3074; `463/543`: **−8.0793 / −0.0793** | 2.667 nats |

**C trailing digits (after prefix `109876`)**

| Backend | `463` (correct) | `543` (wrong) | wrong leads by |
|---|---|---|---|
| b12x | −6.7538 | **−0.0078** | 6.746 nats |
| deep_gemm | −8.0793 | **−0.0793** | 8.000 nats |

**D expression path (prefix `1234567 × 89 = `, deep_gemm only)**: correct −0.6666 (`463` position −1.996), wrong −0.0833 (`543` position −0.246) ⇒ even with the expression prefix, **teacher forcing still prefers `543`, by 1.749 nats**.

## 4. Key counter-evidence: same prefix, different execution path, answer flips

Using the **same 30 ids** (templated ids from render + the ids of `1234567 × 89 = `) sent directly to `/v1/completions` for incremental generation (temp=0, max_tokens=8):

```
generated_text = '109876543'
gen_tokens     = ['109','876','543','<｜end▁of▁sentence｜>']
gen_top1       = {"109": -0.0012, "110": -7.50, "123": -8.00, "108": -9.38, "111": -9.38}
```

That is: **the same preceding text yields `1234567 × 89 = 109876463` when the model writes it token by token (generation), and `109876543` when it is injected externally as a prompt**; teacher-forced scoring on the same prompt agrees with the prefill case (both favour 543, 1.749 nats).

**Conclusion 2**: the distribution at this position (trailing digits after `109876`) **drifts with the execution path**, and the drift (1.749 nats) is enough to flip the choice. The cause is forward shape (one long prefill + 5-token spec-decode batch verification vs single-point `prompt_logprobs` forward). **This is a "silent error" symptom** (same text, same parameters, different scheduling path, different answer), not "the model prefers a wrong answer".

**Retracted wording**: "trailing 543 is the model's inherent arithmetic weakness" **does not hold** — the same model produces 463 on the self-generation path.

## 5. SE-03 verdict (rewritten, publication-ready)

> `SE-03 is a genuine divergence, not a grey zone. The divergence is at the first token: b12x prefers 109 (−0.0032 vs 123 at −5.3797, 5.38 nats), deep_gemm prefers 123 (−1.0280 vs 109 at −3.9030, 2.88 nats); the two backends point in opposite directions and this maps verbatim onto generation (b12x → 109876543, wrong; deep_gemm → 1234567 × 89 = 109876463, correct). Separately, the same stack shows execution-path-dependent numeric drift: with the identical `1234567 × 89 = ` prefix, external prefill (teacher forcing / prefill generation) prefers 543 (by 1.749 nats), while self-generation yields 463.`

Bookkeeping splits into two columns:

1. **kernel-level first-token divergence**: b12x and deep_gemm prefer opposite first tokens for the same prompt (5.38 / 2.88 nats) ⇒ booked to the **backend/kernel account**.
2. **execution-path drift**: same prefix, prefill vs self-generation → 543 / 463 ⇒ booked to the **numeric stability account** (silent-error candidate), mechanism not localised.

## 6. Not evidenced / next steps (by priority)

- **Mechanism not localised**: whether the drift lives in lm_head / last-layer hidden / MoE routing / spec verification is unknown. A minimal isolation experiment (three stack restarts, ~170 s each): (1) disable `--speculative-config`; (2) disable `--enable-prefix-caching`; (3) align `--max-num-seqs` / chunked-prefill. Toggle each alone and see whether 463 / 543 converge.
- Generation was run once per backend (11-13 recorded 4/4 content-identical; not retested this round).
- The slight difference between first-token −0.0032 (b12x) and −0.0054 (11-13) is unattributed (possibly autotune/compile state).
- Whether other prompt types behave the same is untested (only `num_mul` this round).

## 7. Evidence paths

```
101:$BASE/d4/results/se03_tf2_b12x_101.json          # b12x: groups A/B/C
101:$BASE/d4/results/se03_tf2_dgemm_101.json         # deep_gemm: groups A/B/C
101:$BASE/d4/results/se03_tf2_dgemm_formula_101.json # deep_gemm: group D (expression prefix)
101:/tmp/se03_tf2.py                                   # evaluation script (template + positional alignment)
101:/tmp/se03_items2.json /tmp/se03_items_d.json       # A/B/C and D prompt sets
101:/tmp/a6_s14_ab.py                                  # same-id generation vs scoring A/B
101:/tmp/a6_s9_chatprobe.sh /tmp/a6_s10_render.sh      # method verification, template sampling
```

Recompute:
```bash
python3 /tmp/se03_tf2.py --base-url http://127.0.0.1:8888 --model deepseek-v4-flash \
  --items /tmp/se03_items2.json --out <path> --tag <tag>
python3 /tmp/a6_s14_ab.py    # same-id incremental generation vs teacher forcing
```

Bring the stack up (worker first, head 6 s later):
```bash
# 102: bash /tmp/a6_s7_launch.sh worker
# 101: bash /tmp/a6_s7_launch.sh head      # polls startup, about 170 s
# switch backend: bash /tmp/a6_s11_switch.sh head|worker   # b12x ⇄ deep_gemm
```

## 8. Minimal isolation experiment (13:31–13:53, one switch at a time)

Method: for each configuration, restore head/worker scripts from the originals (`*.isoorig`) and delete exactly one flag → force a clean teardown (including releasing master port 25000) → bring the stack up (~170 s) → run `iso_probe.py` (same prompt: 2 chat self-generations + same-id prefill incremental generation + teacher forcing over the two trailing candidates).

| Configuration | chat self-generation | same-id prefill incremental | teacher forcing on trailing digits |
|---|---|---|---|
| baseline (spec on / prefix on / chunk on) | `1234567 × 89 = 109876463` | `109876543` | 543 **−0.0833** vs 463 −0.6666 |
| **spec_off** (drop `--speculative-config`) | **`1234567 × 89 = 109876543`** | `109876543` | 543 −0.0833 vs 463 −0.6666 |
| prefix_off (drop `--enable-prefix-caching`) | `1234567 × 89 = 109876463` | `109876543` | 543 −0.0833 vs 463 −0.6666 |
| chunk_off (drop `--enable-chunked-prefill`) | `1234567 × 89 = 109876463` | `109876543` | 543 −0.0833 vs 463 −0.6666 |

All four configurations ran chat twice and reported `CHAT_SAME` (so this is not request-level randomness); all four `iso_probe` teacher-forced values are **identical to the digit**.

**Conclusion 3 (mechanism localised)**: the only relevant variable is **spec-decode (`method=dspark`, `num_speculative_tokens=5`, `draft_sample_method=probabilistic`)**.
- spec off: chat generation = `109876543`, exactly matching teacher forcing ⇒ **teacher forcing reflects the model's own distribution, and the method is trustworthy**;
- spec on: chat generation = `109876463`, opposite to teacher forcing ⇒ **the spec verification path substitutes the model's preference with another value**;
- `prefix-caching` and `chunked-prefill` are irrelevant to this phenomenon (still 463 when removed).

**Impact (must enter publication wording)**:
1. In SE-03, "deep_gemm answers 109876463 correctly" is **not a more accurate kernel** but a **by-product of spec-decode**; with spec off, deep_gemm also outputs the wrong 109876543.
2. At temp=0 this stack **does not preserve target greedy** — every accuracy/precision evaluation on this stack carries this bias.
3. The first-token divergence (b12x 109 / deep_gemm 123, 5.38 / 2.88 nats) is unrelated to spec: under `spec_off` chat still starts with `1234567 × 89 =` ⇒ **Conclusion 1 stands**.

**Still not evidenced**: only one prompt; where exactly spec fails to preserve (draft sampling vs rejection threshold vs verification batch shape) is not localised; each of the four configurations ran only 2 generations.

Isolation-experiment evidence:
```
101:/tmp/iso_probe.py /tmp/iso_patch.sh /tmp/iso_run.sh /tmp/k2.sh
101:/tmp/iso_spec3.log /tmp/iso_prefix.log /tmp/iso_chunk.log      # raw logs for three configurations
inside container /vllm-workspace/vllm-{head,worker}.sh.isoorig    # original script backups (restored)
```

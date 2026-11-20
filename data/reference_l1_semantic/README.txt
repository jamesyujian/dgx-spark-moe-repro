reference_l1_semantic —— N1 §1.1 L1（DeepGEMM 直连 + 非 ue8m0）的 8 用例原始判定
==========================================================================

本目录 + `scripts/s104_semantic.py` 构成 L1 那一条的完整原件：脚本 + 它在这台机器上的
逐用例判定。此前公告正文只引用了 8 个用例的名字，没有附脚本本体；2026-11-20 从原机上
取回后随包发布，未做任何改写。

来源
----
机    : 116（`james-proart`，1 × RTX PRO 4500 32 GB）
路径  : `~/dgtest/s104_semantic.py`、`~/dgtest/s104_result.json`
取回  : 2026-11-20（脚本 mtime 2026-11-13 22:05）
sha256: 脚本 36d3de6a81673fc1b2502f8ca0f333d572ae2dc3fc4d5c7c7964e2ee2484930e（10,443 B）
        结果 606279055f163886cfb66bdeb0a39b498aeb96c1173543c4532ada608fd5377d（1,495 B）

8 条判定（`s104_result_sm120.json`）
------------------------------------
| # | 用例 | 期望 | 实测 | verdict |
|---|---|---|---|---|
| 1 | bf16 single: A=1,B=1 → K | 128 | mean=128.00 rel=0.00% | ok |
| 2 | bf16 identity: B=I → A | 3.0 | mean=3.000 rel=0.00% | ok |
| 3 | bf16 multi: empty group G2 | G2 不写脏、G3 尾 64 行不写 | `g2_absmax=0` 但 `g3_tail_absmax=128` | **suspicious** |
| 4 | bf16 determinism (10x) | 1 unique hash | 1 unique `f92674f9804a1e90` | ok |
| 5 | fp8 ue8m0: A=1,B=1 → K | 128 | mean=128.00 rel=0.00% | ok |
| 6 | fp8 underflow 1e-30 | 全 0 或有限 | `zero_frac=1.0`, `absmax=0` | ok |
| 7 | fp8 overflow 400*400*128 | 不应 Inf | `inf=0`, `absmax=18874368` | ok |
| 8 | fp8 NON-ue8m0 (disable cast) | 要么正常，要么明确崩；不得静默 NaN | `exit=1`，`RuntimeError: Assertion error (csrc/apis/gemm.hpp:346)` | **error** |

第 8 条就是 N1 §1.1 里「明确崩、不是静默 NaN」的那一行；第 3 条是同一轮里发现的
「空 group 不写脏但尾行写了」的可疑项，公告按原文照录，未改写结论。

怎么复跑
--------
    python3 scripts/s104_semantic.py

前提：目标机要装 **DeepGEMM 2.8.1**（断言出现在 `csrc/apis/gemm.hpp:346`，版本不同行号会变），
并且换上一个 **非 ue8m0 量化的 FP8 权重**；脚本自带一个隔离子进程用例，崩了不会带走主进程。
脚本里没有绝对路径、没有密钥，端口/设备参数在文件头部，按本机改。

没落盘的一半（诚实说明）
------------------------
同一脚本在 **GB10 / SM121** 上的输出（§1.1 第 2 条：`exit=0`、输出全 NaN、无任何 warning）
**没有留下结果文件**：2026-11-20 在 101 / 102 上全盘 `find *s104*`，只有 `s104_deep_gemm.json`
（那是 `deepseek-v4-flash` 的 logprob 采样，不是本脚本）、`s104_flashinfer_b12x.json`、
`~/sm12x/s104*` 下的矩阵脚本，没有 SM121 侧的 semantic 判定。所以索引第 2 行状态仍是
⛔「未定位」，不是「不存在」——正文里那条结论出自 2026-11-13 的现场记录，原件待补跑。

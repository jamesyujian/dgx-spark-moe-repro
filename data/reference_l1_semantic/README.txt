reference_l1_semantic —— N1 §1.1 L1（DeepGEMM 直连 + 非 ue8m0）的 8 用例原始判定
==========================================================================

本目录 + `scripts/s104_semantic.py` 构成 L1 那一条的原件：脚本 + 它逐用例的判定。
此前公告正文只引用了 8 个用例的名字，没有附脚本本体；2026-11-20 从原机取回后随包发布，
未做任何改写。目录内有两份结果：SM120 侧（原始记录）与 SM121 侧（2026-11-20 复跑）。

来源
----
脚本  : 116（`james-proart`，1 × RTX PRO 4500 32 GB），路径 `~/dgtest/s104_semantic.py`
取回  : 2026-11-20（脚本 mtime 2026-11-13 22:05）
sha256: 脚本 36d3de6a81673fc1b2502f8ca0f333d572ae2dc3fc4d5c7c7964e2ee2484930e（10,443 B）

结果 1（`s104_result_sm120.json`，原始记录）
  机    : 116（PRO 4500 32 GB / SM120），DeepGEMM 2.8.1（fork `0f7698fe`）
  sha256: 606279055f163886cfb66bdeb0a39b498aeb96c1173543c4532ada608fd5377d（1,495 B）

结果 2（`s104_result_sm121_dg023post2.json`，2026-11-20 复跑）
  机    : 101（`gx10-james01`，GB10 / SM121，aarch64）
  环境  : 容器 `sglang-sm12x:f832b5c39559`，`deep_gemm 0.2.3.post2`，CUDA 13.0.3
  sha256: 126bd77144b0f3d07391d4fde58b8c4582911c9212405d4e5ae5c01d5a15dec3（1,495 B）
  注意  : **不是同一份库**（2.8.1 vs 0.2.3.post2）。101/102 当日均无 2.8.1 的 aarch64 构建，
          所以这次复跑只能证明「在 0.2.3.post2 上 SM121 怎么失败」，不能直接替代 2.8.1 的现场记录。

8 条判定（`s104_result_sm120.json`，SM120 / 2.8.1）
----------------------------------------------------
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

结果 2 与结果 1 的差异（SM121 / 0.2.3.post2）
**8 条里只有第 8 条的文本不同**，其余 7 条与 SM120 **逐字节相同**（含第 3 条的 `suspicious`、
含第 4 条的 determinism hash `f92674f9804a1e90`）：

| # | 用例 | SM121 / 0.2.3.post2 实测 | verdict |
|---|---|---|---|
| 3 | bf16 multi: empty group G2 | 同上：`g2_absmax=0`、`g3_tail_absmax=128` | **suspicious** |
| 4 | bf16 determinism (10x) | 同上：1 unique `f92674f9804a1e90` | ok |
| 8 | fp8 NON-ue8m0 (disable cast) | `exit=1`，`Assertion error (/deepgemm/csrc/apis/gemm.hpp:415): Unsupported architecture or scaling factor types` | **error** |

两条可独立引用的事实：
  a) **bf16 掩码 GEMM 在 SM120 与 SM121 上逐比特相同**（同一 determinism hash）——两架构在这条
     bf16 路径上没有数值分歧，分歧只出现在 FP8 scale 类型检查上。
  b) **SM121 在 0.2.3.post2 上是"大声失败"**（`:415` 断言，架构/scale 类型不支持），
     不是 N1 §1.1 记录的「静默全 NaN」。也就是：**同一脚本、两个库版本、两种失败模式**。
     「静默」这一步的证据只到版本粒度，公告 §1.1 已在 2026-11-20 按此收紧。

怎么复跑
--------
    python3 scripts/s104_semantic.py

前提：目标机要装 **DeepGEMM**（断言出现在 `csrc/apis/gemm.hpp`，**行号随版本变**：2.8.1 → `:346`，
0.2.3.post2 → `:415`），并且换上一个 **非 ue8m0 量化的 FP8 权重**；脚本自带一个隔离子进程用例，
崩了不会带走主进程。脚本里没有绝对路径、没有密钥，端口/设备参数在文件头部，按本机改。

没落盘的一半（诚实说明，2026-11-20 更新）
-----------------------------------------
§1.1 第 2 条原始表述是「SM121：`exit=0`、输出全 NaN、无任何 warning」（2026-11-13 现场记录）。
该输出**至今没有结果文件**：2026-11-20 在 101 / 102 上全盘 `find *s104*`，只有
`s104_deep_gemm.json`（那是 `deepseek-v4-flash` 的 logprob 采样，不是本脚本）、
`s104_flashinfer_b12x.json`、`~/sm12x/s104*` 下的矩阵脚本。

2026-11-20 的补跑**部分闭合**了这一半：在 101 上用 `deep_gemm 0.2.3.post2` 跑同一脚本，
拿到 8 条 verdict（就是本目录结果 2），但第 8 条是**断言**而非静默 NaN。因此：
  - 「SM121 的 L1 行为」→ 已有可复算产物（结果 2），索引第 2 行由 ⛔ 改为 ○；
  - 「SM121 静默全 NaN」→ **仍无原件**，且当前两台上无 2.8.1 的 aarch64 构建，
    要复现需先在 GB10 上编译 2.8.1（未做）。公告正文已把该句降级为版本相关。

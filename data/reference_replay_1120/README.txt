reference_replay_1120 —— 发布包自解包复跑的原始产物
==================================================

来源
----
- 输入包：sm12x-repro-v1.0.0.zip
  sha256 = 7942b38788d39d4185b43b186566fb6da79892f7d6d2871d88f6aff045fe30c6
- 做法：把 zip 传到机器 → mkdir pkg → unzip 到 pkg（空目录，未触碰原实验目录）→
  直接 run_matrix.sh（同一格 FP8 + triton + TP=2）
- 时间：2026-11-20 08:08:15 起服，08:09:41 收；health 80 s
- 环境：2 × RTX PRO 5000 Blackwell 48 GB，vLLM 0.29.0

结果
----
r1 / r2 / r3 各 18 / 20（acc 0.90），失败题固定 num_mul + logic_weekday；
三轮内容投影 md5 均为 c8db9a77c6d7b58a40df9d66f75344e7，与 2026-11-19 基准逐字节相同；
KV cache 1,355,776 token / 8,192 ctx ⇒ 165.50×；
后端：Selected CutlassFp8BlockScaledMMKernel for Fp8LinearMethod +
      Using TRITON Fp8 MoE backend。

边界（别读错）
--------------
这次跑的是**同一台机器**（原 FP8 实验机），不是第二台机器。
它证明的是：包能从 zip 零解包、路径自洽、无隐藏依赖、结果确定可复现。
跨机（不同 GPU / 不同 vLLM）的 GPU 面复现仍未做，见 README §7.3。

reference_crossmachine_nvfp4 —— 换机（两台不同机型）复跑 NVFP4 格的原始产物
=====================================================================

为什么是这一格
--------------
C3 的基准格是 FP8 + triton + TP=2，需要 **2 张卡 + FP8 权重**。
第二台机器（PRO 4500 / 32 GB / 单卡）两样都没有：单卡装不下 2-way 分片的 FP8 35B，
且它的模型目录里只有 NVFP4。所以**跨机复现只能落在 NVFP4 这一格**（同一支 run_matrix.sh、
同一份 20 题、同一个 qwen3_xml 解析器、同一个 vLLM 0.29.0）。

设置
----
- A 机：2× RTX PRO 5000 Blackwell 48 GB（只用 GPU0，TP=1），vLLM 0.29.0，NVFP4 + `--moe-backend marlin`
- B 机：1× RTX PRO 4500 Blackwell 32 GB（TP=1），vLLM 0.29.0，NVFP4 + `--moe-backend marlin`
- 共同：`--max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization 0.90`，
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml`，20 题各跑 3 轮

结果（这才是要看的）
--------------------
| | r1 | r2 | r3 | 三轮同 md5 |
|---|---|---|---|---|
| A 机（48 G） | 17/20 | 17/20 | 17/20 | 否（74dc1260 / b6fc79cf / b0656b66） |
| B 机（32 G） | 17/20 | 17/20 | **18/20** | 否（74dc1260 / 9d84e813 / f785d08f） |

1. **两机 r1 的内容投影 md5 完全相同**（`74dc12605c34eb0206372f5dcd48f02d`）：
   换机型、换显存容量，首轮 20 题的回答逐字节一致，只有 `latency_s` 不同。
2. **但同一台机器、同一个服务进程内，r2 / r3 与 r1 不等价**：
   A 机 2 题文本变化（`logic_weekday`、`code_bug`）、B 机 3 题（再加 `json_fmt`），
   B 机 r3 的 `code_bug` 还从错判成对（17/20 → 18/20）。
3. 这与 `docs/NOISE_POLICY.md` §2.1 一致：**NVFP4 的差异是内核非确定性，不是采样噪声**；
   本轮在**第二台机型**上独立复现了该结论。

所以：**NVFP4 / marlin 这一格不能当确定性参照**。可作确定性参照的是 FP8 + triton + TP=2 那一格
（三轮 md5 恒为 `c8db9a77…`），而它的跨机证据这里补不出来（硬件/权重条件不满足）。

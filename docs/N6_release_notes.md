# N6 边界白皮书 v1.0 — Release Notes

**三台 SM12x 机器的能干与不能干**：把「哪些能跑、快多少、哪里会停」用可逐条复核的口径固定下来。

- Tag：`v1.1.0`（基线 `v1.0.9` 未改动）
- 范围：3× GB10（SM121，128 G 统一内存）、2× RTX PRO 5000（SM120，48 G/卡）、1× RTX PRO 4500（SM120，32 G）
- 正文：5,487 汉字 / 364 行（中文单语，见「已知缺口」）

## 随包四物件

| 物件 | 路径 | 作用 |
|---|---|---|
| 白皮书正文 | `docs/N6_boundary_whitepaper_v1.0.md` | 十章 + 附录 A/B |
| 证据行 | `docs/N6_evidence.jsonl` | 54 行，字段 `指标/值/等级/机器/镜像/证据路径/采集日期`，可脚本逐条比对 |
| 一键脚本 | `scripts/repro_n6.sh` | 按 digest 拉镜像 → 按本文参数起服 → 跑 `s303_stress.py` 出同口径 csv |
| 口径校验清单 | `docs/N6_release_checklist.md` | 四条硬闸 + 九条禁写 + 12 条映射，可勾选，随包发布 |

配套：`scripts/s303_stress.py`（本次一并入库，此前只在机器上）。

## 三条最硬的结论

1. **单机装不下 155.4 GiB 的 DeepSeek-V4-Flash-DSpark 发行版**。官方已发布的最省形态 155.4 GiB（165.27 B 参数、1 字节/参数、7.63 bit/param 张量本体口径）> 单机可用 121 G，缺口 +34.4 GiB。跨机 PP=2 能跑，但**不存在同权重单机基线**，所以别问「跨机比单机慢多少」——没有分母。
2. **NVFP4 的收益是分机型的**：101 2.47×、102 2.37×、115 1.56×（均为同参对比，115 必带 TP=2·c4）。**TP 1→2 只有 1.13×**，因为 PCIe 无 NVLink。
3. **116 的 13.84 h 静默冻结，机制未定**。原「驱动落后 / 疑似 forward compat」归因已作废——没冻结的那两次 24 h 运行（101 marlin、102 b12x）**自己也带 `CUDA Forward Compatibility mode ENABLED` 标记**。6 h 跑通是正面证据，但 6 h < 13.84 h，**不构成已修复**。

## 口径约定（不复核就别引数）

- 数字只用四档：**已核 / 已核（二手）/ 社区 / 推算**。社区数字一律带「单次、无独立复核、无落盘」。
- 四条硬闸：KV 双值同处（29.75× 与 30.42× 并列）、能效必须写分母（GPU 域 vs 整卡 TBP）、社区三标注、网络口径（27.7 GB/s 是**双向聚合**）。
- 115 的 c1…c16 是 **TP=1 单卡**值（GPU1 全程 0%），不要除以 2。

## 三步复核

```bash
# 1. 逐条比对证据行（不需要读 60 页正文）
jq -r '[.指标,.值,.等级,.机器] | @tsv' docs/N6_evidence.jsonl | column -t -s$'\t'

# 2. 重跑同口径
STAGES="c4:600,c8:600" bash scripts/repro_n6.sh

# 3. 对照自审表
$EDITOR docs/N6_release_checklist.md
```

## 已知缺口（不隐藏）

1. 116 冻结**机制未定**（无 OOM / MCE / AER / Xid，oom-kill 0）。
2. 3 / 4 节点环网：**官方支持、未实测**；CX-7 固件需 ≥1.108.20。
3. 本机内存带宽（规格 273 GB/s）**未测**，只引社区值 207。
4. 白皮书**中文单语**（其余文档 zh/en 对照）；有意不做机器翻译，避免译文误差进入证据链。
5. `c20`、裸 FP8 阈值回归、102 c6 阶跃 —— 均在「停止扩测」边界外，仅登记为缺口。

---

## English summary

**N6 Boundary Whitepaper v1.0 — what three SM12x machines can and cannot do.** A 155.4 GiB DeepSeek-V4-Flash-DSpark release does not fit any single machine here (121 G available, +34.4 GiB shortfall); cross-node PP=2 runs it, but there is no same-weight single-node baseline, so no speedup ratio is claimed. NVFP4 over BF16 is 2.47× / 2.37× / 1.56× per machine (the 1.56× is TP=2, c4). TP 1→2 gives only 1.13× (PCIe, no NVLink). The 13.84 h silent freeze on the PRO 4500 remains **mechanism-unknown**: the "driver lag" attribution was withdrawn because the two 24 h runs that *did not* freeze carry the same `CUDA Forward Compatibility` banner. Package contains the paper, a 54-line machine-checkable `N6_evidence.jsonl`, `scripts/repro_n6.sh`, and a self-audit checklist (4 hard gates + 9 forbidden claims). Paper text is Chinese-only by design.

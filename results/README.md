# results/ —— 跑出来的产物放这里

## 约定

```
results/run_<tag>/
  meta.txt        起服时间、参数、RESULT（HEALTH_OK/EXITED/TIMEOUT）、effective backend、KV 大小
  serve.log       起服全日志（含 Using '<BACKEND>' 那一行）
  r1.json         三轮探针结果（probe_20_prompts.py 产出）
  r2.json
  r3.json
```

`<tag>` 必须能自解释：`fp8_tp2_triton`、`nvfp4_tp1_marlin`、`nvfp4_tp2_b12x_32g`、`sm121_bf16_marlin` …
**不要**用 `test1`、`run_new` 这类名字，也不要覆盖别人的目录。

## 一跑合格的三条判据

1. `meta.txt` 里有 `RESULT=HEALTH_OK` **且** 有 `Using '<BACKEND>'` 行（否则"跑通了但不知道跑的哪个后端"）；
2. `r{1,2,3}.json` 的 `summary.total == 20`、`syntax_invalid == 0`（否则是解析问题，不是精度问题）；
3. 内容投影 md5 记下来（口径见 `data/fingerprints/expected.md5`），十轮才能判"确定/不确定"。

## 不要放进来的东西

- 模型权重（几十 GB，放 `$MODEL_DIR`）；
- 别人的原始实验目录（只放**你这一跑**的产物；要引用历史产物，写进 `docs/EVIDENCE_INDEX.md` 的路径表）；
- 含机器名 / 内网 IP / 绝对家目录的日志（发布前须替换为 `$BASE`）。

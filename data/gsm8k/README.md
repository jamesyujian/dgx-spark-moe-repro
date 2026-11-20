# GSМ8K 1319 子集：从哪来、怎么切、许可

## 1. 来源与许可

- 上游：**GSM8K**（OpenAI，MIT license）——`openai/gsm8k`，`main` split 的 `test` 部分。
- 本包**不再分发全量数据**，只提供：
  1. 取数/切片脚本（见下）；
  2. 切片后的**文件指纹** `md5 = f1203b1ad1e2687fbb4c3d3bc1c6ae84`、`1319` 条；
  3. 每题的判分口径。
- 若你所在司法辖区对再分发有额外限制，请只用脚本自取。

## 2. 切片口径（必须与基准一致，否则计数不可比）

```bash
# 取 GSM8K test，保留前 1319 条（= test 全量），输出 jsonl，字段 {idx, question, answer}
python3 - <<'PY'
import json, pathlib
src = json.load(open("gsm8k_test.json", encoding="utf-8"))   # 自取
out = pathlib.Path("data/gsm8k/gsm8k_test_1319.jsonl")
with out.open("w", encoding="utf-8") as f:
    for i, r in enumerate(src[:1319]):
        gold = r["answer"].split("####")[-1].strip()
        f.write(json.dumps({"idx": i, "question": r["question"], "gold": gold}, ensure_ascii=False) + "\n")
PY
md5sum data/gsm8k/gsm8k_test_1319.jsonl   # 期望 f1203b1ad1e2687fbb4c3d3bc1c6ae84
```

## 3. 评测口径（`scripts/eval_gsm8k.py`）

- 输出**每行一条**：`{idx, verdict: "correct"|"wrong", pred, gold, latency_s}`；
- **计数字段是 `verdict` 字符串，不是布尔 `correct`** —— 统计时用 `verdict == "correct"` 计数，用错字段会得到 0；
- 抽取规则：取最后一个 `####` 后的数字（没有则取正文最后一个数字）；
- `syntax_invalid` 必须为 0 才算这一跑有效（否则是解析问题，不是精度问题）。

## 4. 已知的"看起来怪但正常"的数

- BF16 同栈两跑（`run2` / `run3`）**md5 完全相同**、计数同为 1120 ⇒ 可作参照；
- 20 题集里 `num_mul` 在 BF16 上也错 ⇒ 该题不是量化掉点证据；
- GSM8K 单跑标准误 ≈ **1.0 pp**（1319 题），所以 Δ 小于 ~2 pp 时基本只能写"不可区分"，除非做逐题配对检验。

# s104b 20 题固定集（判分口径说明）

文件：`s104b_prompts.json`（sha256 `2edd60165bac3f9ce48e2e93f70a1a940af26b98b85d32a60638f3d132a963bd`）

## 1. 结构

```json
{ "prompts": [
  { "id": "num_mul", "cat": "numeric", "prompt": "...", "judge": {"type":"num","expect":109876532,"tol":0.01},
    "max_tokens": 128, "tools": null },
  ...
  { "id": "tc_single", "cat": "tool_calling", "prompt": "...",
    "judge": {"type":"toolcall","expect":[{"name":"get_weather","args_any":["北京"]}]},
    "max_tokens": 100, "tools": ["get_weather","get_time"] },
] }
```

**只有 `tc_single` / `tc_multi` / `tc_none` 三题带 `tools` 字段**；`probe_20_prompts.py` 只在 prompt 有 `tools` 时才下发 tools（否则请求里不带 tools，判分会把 `tc_*` 判错）。

## 2. 题型与判分器

| judge.type | 判什么 | 题 |
|---|---|---|
| `num` | 回答里出现某个数（带 tol） | `num_mul`、`num_float`、`num_pct`、`num_unit`、`long_temp`、`long_count`、`bat_ball` |
| `contains` | 命中任一关键词 | `zh_en`、`en_zh`、`fact_capital`、`logic_syllogism` |
| `regex` | 正则命中（代码块、JSON 结构） | `code_rev`、`code_bug`、`code_output`、`json_fmt`（`json_nested` 用结构化判） |
| `json_flat` / `json_nested` | 解析出 JSON 后逐字段比 | `json_fmt`、`json_nested` |
| `toolcall` | **API 层 `tool_calls`** 里是否有指定 name + 参数关键词 | `tc_single`、`tc_multi` |
| `no_toolcall_and_num` | 不该调工具、且数值对 | `tc_none` |

## 3. 三个必须知道的口径坑

1. **`toolcall` 判的是 `tool_calls` 字段，不是文本。** 若服务端 parser 不匹配（例如 Qwen3 方言 + `hermes`），`tool_calls` 会是 `null`、内容退回纯文本；脚本此时用 XML fallback 兜底并置 `parsed_from_content=True`。**统计 API 层可用性时，必须把 `parsed_from_content=True` 的那部分剔除。**
2. **`max_tokens` 是按题给的（100/160/40/128）。** 该模型会先输出一段"thinking"再给答案，token 给小了会 `finish_reason=length`、正文被截断判错 —— 这不是精度问题，是预算问题。**比较不同栈时，max_tokens 必须一致。**
3. **`num_mul` 是一道"模型固有错"题**（正确值 `109876563`，各栈常给 `109876532` / `109876563` / 第三种）。**不要把它当"量化掉点"证据**，它在 BF16 上同样错。

## 4. 复算口径

- 通过率 = `summary.passed / summary.total`；
- 稳定性 = 十轮**内容投影 md5**（`(id, content, tool_calls)` 三元组排序后 md5）是否全同；
- 翻转题 = 十轮里 `(content, tool_calls)` 出现 ≥2 种取值的题；
- Δlp = 同一题在 10 轮上的 logprob 极差（需服务端返回 logprobs，若未返回则该字段缺省）。

# A6 · SE-03 teacher-forcing 判定（deep_gemm / b12x 双侧对照）

English：[`SE-03_teacher_forcing_verdict.en.md`](SE-03_teacher_forcing_verdict.en.md)（本文件为准）

> **2026-11-20 修订（v1.0.8）——本文的机制结论已被 P4 改判，先读这段再读正文**
> 11-19 的判定「唯一相关变量是 spec-decode」在**该题文字结果**这一层成立；但 P4（11-20，101+102 TP=2，18:00 复核）把层拆开后改判：
> - **同配置同会话 rep 间 Δ=0.0000**（100 对、确定）；**`b12x` 与 `deep_gemm` 的 prefill 分布 20/20 题差 >0.1 nats（中位 0.363、0.128–1.396）**，答案邻位 gen 20/20（中位 0.525）⇒ 主因在 **MoE GEMM 后端**，不是 spec；
> - spec 只是在文字层叠加（≈0.05 nats，并换 top-20 成员）；
> - 另发现 **`deep_gemm` 同配置跨起服在两道乘法题上漂 ≈0.7 nats、`b12x` 不漂** ⇒ 机制**未定**，只能写现象。
> ⇒ 机制定为「**kernel 主因 + spec 叠加**」；**结论 3 与影响 1 已按此改写**（见 126、132 行），其余实验与数字保留。**边界**：样本 20 题 ×5 rep（四族 6/5/3/6），logit 层四族全中；**首 token 20/20 相同，但 16-token 文本 13/20 相同、7/20 在 16 token 内就分叉（A1、A2、A4、B4、C2、C3、D1，含数值不同的答案）** ⇒「跨族普遍、且会落到输出」可写，「某族更准 / 全量都会变」不可写；机制直接取证未做。

- 日期：2026-11-19
- 主机：**101（GB10 / SM121）head + 102 worker**，TP=2 / `--nnodes 2`，served-model `deepseek-v4-flash`，port 8888
- 栈：`anemll/dspark-vllm-gx10:0.1.1`，DeepSeek-V4-Flash-DSpark，`quantization=deepseek_v4_fp8`（MoE 专家 mxfp4），`--speculative-config {"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}`、`--enable-prefix-caching`、`kv-cache-dtype nvfp4_ds_mla`
- 题：`计算 1234567 × 89 的结果，只输出数字，不要解释。`；正确值 **109876463**；SE-03 记录的 b12x 错值为 **109876543**

## 1. 方法（先解决口径，再谈数字）

两个前置坑都实测踩过，写法固定在 `se03_tf2.py`：

1. **raw `/v1/completions` 的 prompt 不能直接喂原文**：该栈会把原文当“文档续写”，续出 `"\n}<｜end▁of▁file｜>`，此时 `prompt_logprobs` 的绝对值全部作废（首 token 会给出 −16.24 这类与 11-13 差三个数量级的数）。
2. **必须用 `/v1/chat/completions/render` 取模板化后的 `token_ids`**（本题 22 个：`[0,128803,4339,223,6895,18009,25,2863,223,4362,223,17488,303,1299,10626,8283,303,4916,9068,320,128804,128822]`，末位 128822 = assistant 起始），再以 **id 列表**作 `prompt` 送 `/v1/completions`（`max_tokens=1`、`prompt_logprobs=1`）。
3. **`prompt_logprobs[0]` 恒为 null（BOS）**，必须按位置对齐（`prompt_logprobs[i]` ↔ 第 i 个 token）；`max_tokens=0` 会被 400 拒（"must be at least 1"）。
4. 候选 token 数用 `/tokenize` 精确取得，候选得分 = 其在末尾几位的 logprob 均值。

**方法自证**：同一法在 b12x 上得首 token `109` = **−0.0032**（11-13 记录 −0.0054，同量级）；在 deep_gemm 上得 `123` = **−1.0280**（11-13 记录 −1.028，**逐位吻合**）。

## 2. 首 token（A 组）——两个后端偏好相反

| 后端 | `109` | `123` | 偏好 | 领先幅度 |
|---|---|---|---|---|
| `flashinfer_b12x` | **−0.0032** | −5.3797 | **109** | 5.3765 nats |
| `deep_gemm` | −3.9030 | **−1.0280** | **123** | 2.8750 nats |

**结论 1（主判据）**：SE-03 的分歧出现在**首 token**，且两后端方向相反、幅度 5.38 / 2.88 nats —— **不是近邻平手、不是采样噪声**。首 token 选择直接决定后续路径：选 `109` ⇒ 走“直接给数”，选 `123` ⇒ 走“先写算式”。

**生成期实测与之一致**（双方各 1 次，temp=0）：

| 后端 | 生成输出 | 首 token 归属 | 判分 |
|---|---|---|---|
| b12x | `109876543` | 109（错路径） | 错（差 80） |
| deep_gemm | `1234567 × 89 = 109876463` | 123（算式路径） | 对 |

## 3. 整答与尾数（B / C 组）——两后端同向，但与生成结果打架

**B 整答（平均 token logprob）**

| 后端 | 正确 `109876463` | 错误 `109876543` | 逐位 lp（正确 / 错误） | 错误领先 |
|---|---|---|---|---|
| b12x | −2.0048 | **−0.0067** | `109`: −0.0053 / −0.0048；`876`: −0.0019 / −0.0073；`463/543`: **−6.0071 / −0.0078** | 1.998 nats |
| deep_gemm | −4.0966 | **−1.4299** | `109`: −3.903 / −3.903；`876`: −0.3074 / −0.3074；`463/543`: **−8.0793 / −0.0793** | 2.667 nats |

**C 尾数（前缀 `109876` 之后）**

| 后端 | `463`（正确） | `543`（错误） | 错误领先 |
|---|---|---|---|
| b12x | −6.7538 | **−0.0078** | 6.746 nats |
| deep_gemm | −8.0793 | **−0.0793** | 8.000 nats |

**D 算式路径（前缀 `1234567 × 89 = `，仅 deep_gemm）**：正确 −0.6666（`463` 位 −1.996）、错误 −0.0833（`543` 位 −0.246）⇒ 即便给了算式前缀，**teacher forcing 仍偏 `543`，领先 1.749 nats**。

## 4. 关键反证：同一前缀、换执行路径，答案翻转

用**同一批 30 个 id**（render 后的模板 id + `1234567 × 89 = ` 的 id）直接送 `/v1/completions` 做增量生成（temp=0、max_tokens=8）：

``
generated_text = '109876543'
gen_tokens     = ['109','876','543','<｜end▁of▁sentence｜>']
gen_top1       = {"109": -0.0012, "110": -7.50, "123": -8.00, "108": -9.38, "111": -9.38}
``

即：**同一段前置文本，由模型自己逐 token 写出（生成）时得到 `1234567 × 89 = 109876463`，由外部把它作为 prompt 预填入时得到 `109876543`**；而 teacher-forced 打分（同 prompt）与预填情形一致（都偏 543，1.749 nats）。

**结论 2**：该位置（`109876` 之后的尾数选择）的分布**随执行路径漂移**，且漂移量（1.749 nats）足以翻转选择。差别来自前向形状（整段 prefill + spec-decode 5-token 批量校验 vs 单点 `prompt_logprobs` 前向）。**这是“静默错误”类指征**（同文本、同参、不同调度路径给出不同答案），不是“模型偏好错答案”。

**因此撤回旧表述**：“尾数 543 是模型固有算术弱点”**不成立**——同一模型在自生成路径下给出 463。

## 5. SE-03 判定（改写版，可直接发布）

> `SE-03 属真分歧，不是灰区。分歧点在首 token：b12x 偏 109（−0.0032 vs 123 −5.3797，5.38 nats）、deep_gemm 偏 123（−1.0280 vs 109 −3.9030，2.88 nats），两后端方向相反，且与生成输出逐字对应（b12x → 109876543 错；deep_gemm → 1234567 × 89 = 109876463 对）。另发现同栈存在执行路径相关的数值漂移：同一 `1234567 × 89 = ` 前缀，外部预填前进（teacher forcing / 预填生成）偏 543（领先 1.749 nats），模型自生成路径却给出 463。`

记账拆两栏：

1. **kernel 级首 token 分歧**：b12x 与 deep_gemm 对同一 prompt 的首 token 偏好相反（5.38 / 2.88 nats）⇒ 记在**后端/kernel 账**。
2. **执行路径漂移**：同前缀、预填 vs 自生成 → 543 / 463 ⇒ 记在**数值稳定性账**（静默错误候选），机制未定位。

## 6. 未取证 / 下一步（按优先级）

- **机制未定位**：漂移发生在 lm_head / 末层 hidden / MoE 路由 / spec 校验，未定位。最小隔离实验（3 次起栈，每次 ~170 s）：① `--speculative-config` 关；② `--enable-prefix-caching` 关；③ `--max-num-seqs`/chunked-prefill 对齐。三者单独关，看 463/543 是否收敛。
- 生成期只各跑 1 次（11-13 曾记 4/4 内容一致，本轮未复测）。
- 首 token 绝对值 b12x −0.0032 与 11-13 −0.0054 的轻微差未归因（疑 autotune/编译态）。
- 未测其它题型是否同型（本轮只 `num_mul` 一题）。

## 7. 证据路径

``
101:$BASE/d4/results/se03_tf2_b12x_101.json          # b12x：A/B/C 三组
101:$BASE/d4/results/se03_tf2_dgemm_101.json         # deep_gemm：A/B/C 三组
101:$BASE/d4/results/se03_tf2_dgemm_formula_101.json # deep_gemm：D 组（算式前缀）
101:/tmp/se03_tf2.py                                   # 评测脚本（模板+位置对齐）
101:/tmp/se03_items2.json /tmp/se03_items_d.json       # A/B/C、D 题集
101:/tmp/a6_s14_ab.py                                  # 同 id 生成 vs 打分 A/B
101:/tmp/a6_s9_chatprobe.sh /tmp/a6_s10_render.sh      # 口径校验、模板取样
``

复算：
``bash
python3 /tmp/se03_tf2.py --base-url http://127.0.0.1:8888 --model deepseek-v4-flash \
  --items /tmp/se03_items2.json --out <path> --tag <tag>
python3 /tmp/a6_s14_ab.py    # 同 id 增量生成 vs teacher forcing
``

起栈（先 worker 后 head，间隔 6 s）：
``bash
# 102: bash /tmp/a6_s7_launch.sh worker
# 101: bash /tmp/a6_s7_launch.sh head      # 轮询 startup，约 170 s
# 切后端: bash /tmp/a6_s11_switch.sh head|worker   # b12x ⇄ deep_gemm
``

## 8. 最小隔离实验（13:31–13:53，每次只改一个开关）

方法：每档把 head/worker 脚本还原到原始（`*.isoorig`）后只删一个 flag → 强制清栈（含释放 master 端口 25000）→ 起栈（约 170 s）→ 跑 `iso_probe.py`（同一题：chat 自生成 2 次 + 同 id 预填增量生成 + teacher forcing 尾数两候选）。

| 配置 | chat 自生成 | 同 id 预填增量生成 | teacher forcing 尾数 |
|---|---|---|---|
| baseline（spec on / prefix on / chunk on） | `1234567 × 89 = 109876463` | `109876543` | 543 **−0.0833** vs 463 −0.6666 |
| **spec_off**（删 `--speculative-config`） | **`1234567 × 89 = 109876543`** | `109876543` | 543 −0.0833 vs 463 −0.6666 |
| prefix_off（删 `--enable-prefix-caching`） | `1234567 × 89 = 109876463` | `109876543` | 543 −0.0833 vs 463 −0.6666 |
| chunk_off（删 `--enable-chunked-prefill`） | `1234567 × 89 = 109876463` | `109876543` | 543 −0.0833 vs 463 −0.6666 |

四次 chat 各跑 2 次均 `CHAT_SAME`（非请求级随机）；四次 `iso_probe` 的 teacher-forced 值**逐位相同**。

**结论 3（机制定位，2026-11-20 改判）**：11-19 的「唯一相关变量 = spec-decode」只在**该题文字结果**这一层成立。P4 把层拆开后：**主因是 MoE GEMM 后端**——同配置同会话 rep 间 0.0000（100 对，确定）；**`b12x` 与 `deep_gemm` 的 prefill 分布 20/20 题差 >0.1 nats（中位 0.363、0.128–1.396）**，答案邻位 gen 20/20（中位 0.525）；**spec 只叠加**：关/开 spec 在 11/20 题 prefill 完全相同、余 9 题 ≤0.49，且 spec 臂自身 rep 间漂 0.7063（唯一非确定源）⇒ 机制定为「**kernel 主因 + spec 叠加**」，旧口径作废。
- 关掉 spec：chat 生成 = `109876543`，与 teacher forcing 一致 ⇒ **teacher forcing 反映模型自身分布，方法可信**（这一条不变）；
- 开着 spec：chat 生成 = `109876463`，与 teacher forcing 相反 ⇒ **spec 校验路径把模型偏好替换成另一个值**（这一条也不变；但**它只解释文字层，不解释两侧 logit 分布本就不同**）；
- `prefix-caching`、`chunked-prefill` 均与本现象无关（删掉后仍 463）。

**影响（必须写进发布口径）**：
1. SE-03 里"deep_gemm 答对 109876463"**既不能写成 kernel 更准，也不能只归因于 spec**：spec 是叠加项，主因在 GEMM 后端层的分布差（spec 关掉后两侧 logit 仍差 0.36–0.61 nats，只是该题文字恰好一致）。⇒ 对外只写「**该题上 kernel 层分布差可复现**」，不写"谁更准"。
2. 该栈在 temp=0 下**不保真 target greedy** —— 凡用此栈做的正确率/精度评测都带此项偏差。
3. 首 token 分歧（b12x 109 / deep_gemm 123，5.38 / 2.88 nats）与 spec 无关：`spec_off` 下 chat 仍以 `1234567 × 89 =` 开头 ⇒ **结论 1 不受影响**。

**仍未取证（2026-11-20 23:00 更新）**：11-19 只有 1 题；**P4-c 已扩到 20 题 ×5 rep、四族全覆盖（A 6/6、B 5/5、C 3/3、D 6/6 logit 全中）**；**首 token 20/20 相同，16-token 文本仅 13/20 相同、7/20 在 16 token 内就分叉（含数值不同的答案）** ⇒ 「差会落到输出」可以写，「全量题集都会变 / 某族更准」不能写；**机制仍无直接取证**（未做 kernel 选择/调优落点的受控切换，`deep_gemm` 跨起服漂 ≈0.7 只是现象）；spec 不保真的具体位置（draft 采样 vs 拒绝阈值 vs 校验批形状）未定位。

隔离实验证据：
``
101:/tmp/iso_probe.py /tmp/iso_patch.sh /tmp/iso_run.sh /tmp/k2.sh
101:/tmp/iso_spec3.log /tmp/iso_prefix.log /tmp/iso_chunk.log      # 三档原始日志
容器内 /vllm-workspace/vllm-{head,worker}.sh.isoorig               # 原始脚本备份（已还原）
``

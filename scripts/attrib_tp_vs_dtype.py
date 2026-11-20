#!/usr/bin/env python3
# A2 替代归因分析（2026-11-19）
# 输入：NVFP4@TP2 十轮 s104h_nvfp4_tp2_r1..10.json；NVFP4@TP1 十轮 s104i_base_marlin_r1..10.json
# 输出：跨轮翻转题数、唯一轨迹数、Δlp、与 TP1 集合的差异（判断 TP 是否为非确定性来源）
import glob, json, os, sys, statistics

# 原始实验目录：用 BASE 环境变量指定（默认 ~/sm12x），不再硬编码绝对路径
BASE = os.environ.get("BASE", os.path.expanduser("~/sm12x"))

def load(pat, tag):
    files = sorted(glob.glob(pat), key=lambda p: int(''.join(c for c in p.split('_r')[-1] if c.isdigit())))
    runs = []
    for f in files:
        d = json.load(open(f))
        m = {r['id']: r for r in d['results']}
        runs.append((f.split('/')[-1], m))
    return runs

def analyze(name, runs):
    ids = list(runs[0][1].keys())
    print(f"\n## {name}  （{len(runs)} 轮 / {len(ids)} 题）")
    flipped, per_q = [], {}
    for q in ids:
        contents = [m[q]['content'] for _, m in runs if q in m]
        seqs = [tuple(t['tok'] for t in m[q]['tokens']) for _, m in runs if q in m]
        lps = [[t['lp'] for t in m[q]['tokens']] for _, m in runs if q in m]
        uniq_c = len(set(contents))
        uniq_s = len(set(seqs))
        dlp = None
        if uniq_s > 1:
            base = lps[0]
            worst = 0.0
            for other in lps[1:]:
                n = min(len(base), len(other))
                worst = max(worst, max(abs(base[i] - other[i]) for i in range(n)))
            dlp = round(worst, 3)
        per_q[q] = (uniq_c, uniq_s, dlp)
        if uniq_s > 1:
            flipped.append(q)
    print(f"- 跨轮翻转题数：**{len(flipped)} / {len(ids)}**" + (f" → {flipped}" if flipped else "（十轮逐字节全同）"))
    print(f"- 唯一轨迹数分布：" + ", ".join(f"{q}:{v[1]}" for q, v in per_q.items() if v[1] > 1) or "- 全 1")
    mx = [(q, v[2]) for q, v in per_q.items() if v[2] is not None]
    if mx:
        print(f"- 最大 Δlp：**{max(m[1] for m in mx):.3f}**（{max(mx, key=lambda x: x[1])[0]}）")
    # 内容层面的跨轮翻转（含标点差异）
    cf = [q for q in ids if per_q[q][0] > 1]
    print(f"- 内容层面翻转：{len(cf)} / {len(ids)}")
    return set(flipped)

t2 = load(os.path.join(BASE, 'd4/results/s104h_nvfp4_tp2_r*.json'), 'tp2')
t1 = load(os.path.join(BASE, 'd4/results/s104i_base_marlin_r*.json'), 'tp1')
f2 = analyze('NVFP4 @ TP=2（115 双卡，marlin，2026-11-19）', t2)
f1 = analyze('NVFP4 @ TP=1（115 单卡，marlin，2026-11-18）', t1)
print(f"\n## TP 归因")
print(f"- TP1 翻转集：{sorted(f1)}")
print(f"- TP2 翻转集：{sorted(f2)}")
print(f"- 交集：{sorted(f1 & f2)}；仅 TP1：{sorted(f1 - f2)}；仅 TP2：{sorted(f2 - f1)}")
print(f"- 判定：{'两档都非确定 ⇒ 非确定性不随 TP 出现/消失，TP 不是来源' if (f1 and f2) else '需复核'}")

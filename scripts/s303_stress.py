#!/usr/bin/env python3
"""S3-03 长稳压测客户端（GB10/SM121）。

口径与 eval_gsm8k.py 对齐：temperature=0, seed=0, max_tokens=512,
stop=["\\nQuestion:", "\\nQ:"], enable_thinking=False。
多并发档顺序执行，每 5 s 采一次 nvidia-smi（温度/功耗/显存/利用率），
按窗口聚合请求数、错误数、吞吐 tok/s、平均与最大延迟。

用法：
  python3 s303_stress.py --port 8201 --model qwen36-bf16 \
      --stages "c4:14400,c6:14400" --out-dir ~/sm12x/results3/stress
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request


def load_questions(path):
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                qs.append(json.loads(line)["question"])
            except Exception:
                continue
    return qs


def unified_mem_mib():
    """GB10 统一内存：nvidia-smi 的 memory.used 返回 [N/A]，
    改用 /proc/meminfo 的 MemTotal-MemAvailable 作为占用口径。"""
    try:
        vals = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                k, v = line.split(":", 1)
                vals[k] = float(v.strip().split()[0])
        used_kb = vals["MemTotal"] - vals.get("MemAvailable", vals["MemFree"])
        return round(used_kb / 1024.0, 1)
    except Exception:
        return None


def gpu_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,power.draw,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        p = [x.strip() for x in out.split(",")]

        def f(x):
            try:
                return float(x)
            except Exception:
                return None

        mem = f(p[2])
        if mem is None or mem <= 0:
            mem = unified_mem_mib()
        return dict(temp_c=f(p[0]), power_w=f(p[1]), mem_used_mib=mem, util_pct=f(p[3]))
    except Exception:
        return dict(temp_c=None, power_w=None, mem_used_mib=unified_mem_mib(), util_pct=None)


class Counter:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.reqs = 0
        self.errs = 0
        self.tokens = 0
        self.lat_sum = 0.0
        self.lat_max = 0.0

    def add(self, ok, tokens, lat):
        with self.lock:
            if ok:
                self.reqs += 1
                self.tokens += tokens
                self.lat_sum += lat
                if lat > self.lat_max:
                    self.lat_max = lat
            else:
                self.errs += 1

    def pop(self):
        with self.lock:
            s = (self.reqs, self.errs, self.tokens, self.lat_sum, self.lat_max)
            self.reset()
        return s


def request_one(url, model, question, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "stop": ["\nQuestion:", "\nQ:"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    lat = time.time() - t0
    tok = (body.get("usage") or {}).get("completion_tokens", 0) or 0
    return lat, tok


def worker(cid, stop_at, state, url, model, qs, max_tokens, timeout, counter, err_detail):
    i = cid
    while time.time() < stop_at and not state["stop"]:
        q = qs[i % len(qs)]
        i += 1
        try:
            lat, tok = request_one(url, model, q, max_tokens, timeout)
            counter.add(True, tok, lat)
        except Exception as e:
            counter.add(False, 0, 0.0)
            if len(err_detail) < 30:
                err_detail.append(repr(e)[:200])


def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return dict(min=None, max=None, mean=None, p50=None)
    return dict(min=round(min(vals), 3), max=round(max(vals), 3),
                mean=round(sum(vals) / len(vals), 3),
                p50=round(statistics.median(vals), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stages", required=True, help='如 "c4:14400,c6:14400"')
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data", default="/home/james/sm12x/data/gsm8k_test_1319.jsonl")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--interval", type=int, default=5)
    args = ap.parse_args()

    qs = load_questions(args.data)
    if not qs:
        raise SystemExit("no questions loaded")
    url = "http://127.0.0.1:%d/v1/chat/completions" % args.port
    os.makedirs(args.out_dir, exist_ok=True)

    stages = []
    for part in args.stages.split(","):
        name, sec = part.split(":")
        stages.append((name, int(sec)))

    counter = Counter()
    state = {"stop": False}
    err_detail = []
    all_summaries = []

    for stage_name, dur in stages:
        rows = []
        stage_start = time.time()
        stop_at = stage_start + dur
        conc = int(stage_name.lstrip("c")) if stage_name.startswith("c") else 1

        print("[stage %s] conc=%d dur=%ds start=%s" % (
            stage_name, conc, dur,
            time.strftime("%H:%M:%S", time.localtime(stage_start))), flush=True)

        threads = []
        for cid in range(conc):
            t = threading.Thread(target=worker, daemon=True,
                                 args=(cid, stop_at, state, url, args.model, qs,
                                       args.max_tokens, args.timeout, counter, err_detail))
            t.start()
            threads.append(t)

        stage_tokens = 0
        csv_path = os.path.join(args.out_dir, "stress_%s.csv" % stage_name)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "stage", "conc", "temp_c", "power_w", "mem_used_mib",
                        "util_pct", "reqs", "errs", "tok_s", "avg_lat_s", "max_lat_s"])
            prev = time.time()
            while True:
                now = time.time()
                if now >= stop_at:
                    break
                time.sleep(max(0.0, min(now + args.interval, stop_at) - now))
                now2 = time.time()
                win = max(now2 - prev, 1e-6)
                prev = now2
                reqs, errs, toks, lat_sum, lat_max = counter.pop()
                stage_tokens += toks
                g = gpu_sample()
                row = dict(
                    ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    stage=stage_name, conc=conc,
                    temp_c=g["temp_c"], power_w=g["power_w"],
                    mem_used_mib=g["mem_used_mib"], util_pct=g["util_pct"],
                    reqs=reqs, errs=errs,
                    tok_s=round(toks / win, 2),
                    avg_lat_s=round(lat_sum / reqs, 3) if reqs else None,
                    max_lat_s=round(lat_max, 3) if reqs else None,
                )
                w.writerow([row[k] for k in ["ts", "stage", "conc", "temp_c", "power_w",
                                             "mem_used_mib", "util_pct", "reqs", "errs",
                                             "tok_s", "avg_lat_s", "max_lat_s"]])
                rows.append(row)
                if len(rows) % 60 == 0:
                    print("  [%s] t=%.0fmin temp=%s power=%s tok_s=%s reqs=%d errs=%d" % (
                        stage_name, (time.time() - stage_start) / 60,
                        row["temp_c"], row["power_w"], row["tok_s"], row["reqs"], row["errs"]),
                        flush=True)

        for t in threads:
            t.join(timeout=60)

        summary = dict(
            stage=stage_name, conc=conc, duration_s=int(time.time() - stage_start),
            windows=len(rows),
            temp_c=stat([r["temp_c"] for r in rows]),
            power_w=stat([r["power_w"] for r in rows]),
            mem_used_mib=stat([r["mem_used_mib"] for r in rows]),
            util_pct=stat([r["util_pct"] for r in rows]),
            tok_s=stat([r["tok_s"] for r in rows]),
            avg_lat_s=stat([r["avg_lat_s"] for r in rows]),
            total_reqs=sum(r["reqs"] for r in rows),
            total_errs=sum(r["errs"] for r in rows),
            total_tokens=stage_tokens,
        )
        all_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)
        counter.pop()

    sp = os.path.join(args.out_dir, "stress_summary.json")
    json.dump(dict(port=args.port, model=args.model, stages=all_summaries,
                   errors_sample=err_detail[:30],
                   finished_at=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("SUMMARY ->", sp, flush=True)


if __name__ == "__main__":
    main()

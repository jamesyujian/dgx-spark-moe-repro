#!/usr/bin/env python3
"""S1-04-b runner: 20-prompt fixed set with tool calling, deterministic judging.

Usage:
  python3 s104b_run.py --base-url http://127.0.0.1:8201 --model qwen36-35b-nvfp4 \
      --tag marlin_gpu1 --out results/run_marlin_gpu1/r1.json [--repeat 2]
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error

PROMPTS_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prompts", "s104b_prompts.json")

FILLER = ("这是一段用于测试长上下文定位能力的填充文本，内容与问题无关。"
          "运维记录里提到过多台设备、多个时间窗口和若干例行检查项，"
          "但只有一处包含我们需要的事实。请忽略无关描述。")

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询指定城市的当前天气",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名，例如 北京"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "获取当前时间",
        "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}}}},
]


def build_long_text(needle):
    """needle 放在约 3-4 千字中的中段，前后各有大量无关段落。"""
    parts = []
    for i in range(18):
        parts.append(f"【段落 {i+1}】{FILLER}")
    parts.insert(9, f"【关键记录】{needle}")
    return "\n\n".join(parts)


LONG_TEXTS = {
    "LONG_TEXT_TEMP": build_long_text("巡检单第 7 项：机柜 C 的温度读数为 45 摄氏度，其余机柜均在 30 度上下。"),
    "LONG_TEXT_COUNT": build_long_text("监控面板显示，今日的告警计数为 3 次，全部为磁盘温度瞬时告警，已自动恢复。"),
}


def post(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def extract_json(text):
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def numbers(text):
    out = []
    for tok in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        try:
            out.append(float(tok.replace(",", "")))
        except Exception:
            pass
    return out


def parse_xml_toolcalls(content):
    """Fallback: Qwen-style <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>."""
    calls = []
    for fn, body in re.findall(r"<function=([^>\s]+)>(.*?)</function>", content or "", re.S):
        args = {}
        for k, v in re.findall(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", body, re.S):
            args[k.strip()] = v.strip()
        calls.append({"function": {"name": fn.strip(), "arguments": json.dumps(args, ensure_ascii=False)}})
    return calls


def resolve(obj, path):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except Exception:
                return None
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def judge(spec, content, tool_calls):
    t = spec.get("type")
    content = content or ""
    if t == "num":
        exp = float(spec["expect"]); tol = float(spec.get("tol", 0.01))
        hit = any(abs(n - exp) <= tol for n in numbers(content))
        return hit, f"expect {spec['expect']} got {numbers(content)[:4]}"
    if t == "contains":
        low = content.lower()
        hit = any(k.lower() in low for k in spec["any"])
        return hit, f"expect any {spec['any']}"
    if t == "regex":
        hit = re.search(spec["pattern"], content, re.S) is not None
        return hit, f"pattern {spec['pattern'][:40]}"
    if t == "json_flat":
        obj = extract_json(content)
        ok = isinstance(obj, dict) and all(str(obj.get(k)) == str(v) for k, v in spec["expect"].items())
        return ok, f"json={obj}"
    if t == "json_nested":
        obj = extract_json(content)
        if not isinstance(obj, dict):
            return False, "no json"
        bad = {k: resolve(obj, k) for k, v in spec["expect"].items() if str(resolve(obj, k)) != str(v)}
        return (len(bad) == 0), f"bad={bad}"
    if t == "toolcall":
        names = [c.get("function", {}).get("name") for c in (tool_calls or [])]
        ok = True
        detail = []
        for exp in spec["expect"]:
            if exp["name"] not in names:
                ok = False; detail.append(f"missing {exp['name']}")
                continue
            if exp.get("args_any"):
                raw = ""
                for c in tool_calls:
                    if c.get("function", {}).get("name") == exp["name"]:
                        raw += str(c.get("function", {}).get("arguments", ""))
                if not any(k.lower() in raw.lower() for k in exp["args_any"]):
                    ok = False; detail.append(f"{exp['name']} args miss {exp['args_any']}")
        return ok, f"calls={names} {';'.join(detail)}"
    if t == "no_toolcall_and_num":
        if tool_calls:
            return False, f"unexpected calls={[c['function']['name'] for c in tool_calls]}"
        exp = float(spec["expect"])
        hit = any(abs(n - exp) <= float(spec.get("tol", 0.01)) for n in numbers(content))
        return hit, f"expect {spec['expect']} got {numbers(content)[:4]}"
    return False, f"unknown judge {t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8201")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=PROMPTS_DEFAULT)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma separated ids")
    ap.add_argument("--thinking", action="store_true", help="keep thinking mode (default off)")
    args = ap.parse_args()

    spec = json.load(open(args.prompts, encoding="utf-8"))
    prompts = spec["prompts"]
    if args.only:
        keep = set(args.only.split(","))
        prompts = [p for p in prompts if p["id"] in keep]

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    results = []
    for rep in range(args.repeat):
        for p in prompts:
            q = p["q"]
            for key, val in LONG_TEXTS.items():
                q = q.replace(key, val)
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": q}],
                "temperature": 0,
                "seed": 1234,
                "max_tokens": p.get("max_tokens", 128),
                "chat_template_kwargs": {"enable_thinking": args.thinking},
            }
            if p.get("tools"):
                payload["tools"] = [t for t in TOOLS if t["function"]["name"] in p["tools"]]
                payload["tool_choice"] = "auto"
            t0 = time.time()
            try:
                resp = post(url, payload)
                msg = resp["choices"][0]["message"]
                content = msg.get("content") or ""
                tcs = msg.get("tool_calls") or []
                usage = resp.get("usage", {})
                fallback = False
                if p.get("tools") and not tcs and "<tool_call>" in content:
                    tcs = parse_xml_toolcalls(content)
                    fallback = bool(tcs)
                ok, detail = judge(p["judge"], content, tcs)
                err = None
            except Exception as e:
                content, tcs, usage = "", [], {}
                ok, detail, err = False, "request failed", repr(e)[:300]
            dt = round(time.time() - t0, 3)
            rec = {"id": p["id"], "cat": p["cat"], "rep": rep + 1, "pass": ok, "detail": detail,
                   "error": err, "content": content, "tool_calls": tcs,
                   "parsed_from_content": fallback,
                   "usage": usage, "latency_s": dt}
            results.append(rec)
            print(f"[{args.tag}] rep{rep+1} {p['id']:<18} {'PASS' if ok else 'FAIL'}  {dt:6.2f}s  {content[:60]!r}", flush=True)

    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    by_cat = {}
    for r in results:
        c = by_cat.setdefault(r["cat"], [0, 0])
        c[1] += 1
        c[0] += 1 if r["pass"] else 0
    summary = {"tag": args.tag, "model": args.model, "base_url": args.base_url,
               "prompt_version": spec.get("version"), "repeat": args.repeat,
               "total": total, "passed": passed,
               "acc": round(passed / total, 4) if total else 0,
               "by_cat": {k: {"pass": v[0], "n": v[1]} for k, v in by_cat.items()},
               "gen_tokens": sum(r["usage"].get("completion_tokens", 0) for r in results),
               "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("OUT", args.out)


if __name__ == "__main__":
    main()

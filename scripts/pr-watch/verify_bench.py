#!/usr/bin/env python3
"""校验 bench-result.json：收益是否真的超过噪声，prefill/RSS 是否回退，正确性自检是否通过。

用法：
  verify_bench.py <bench-result.json> --pr 8 --sha b484e250...

退出码：0 通过；1 不通过；2 输入不合法。

判据（阈值都可由命令行/环境覆盖，默认值与 config.env 一致）：
  - A/B 各至少 4 轮（AGENTS §2：≥4 轮轮转）
  - 主指标 decode：gain ≥ max(min_gain, noise_mult × A/A 噪声)，且 B 的最小值 > A 的最大值（组间不重叠）
  - prefill：回退不超过 prefill_max_regression
  - RSS：不超过 rss_max_regression
  - correctness.llama_moe_verify_slot 必须为 true
  - pr / sha 必须与当前评审对象一致，created_at 不超过 max_age_hours
"""
import argparse
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path


def median(xs):
    return statistics.median(xs)


def rel_gain(a, b):
    ma = median(a)
    return (median(b) - ma) / ma if ma else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--pr", type=int, required=True)
    ap.add_argument("--sha", required=True)
    ap.add_argument("--min-gain", type=float, default=0.015)
    ap.add_argument("--noise-mult", type=float, default=1.0)
    ap.add_argument("--min-rounds", type=int, default=4)
    ap.add_argument("--prefill-max-regression", type=float, default=0.02)
    ap.add_argument("--rss-max-regression", type=float, default=0.02)
    ap.add_argument("--max-age-hours", type=float, default=72)
    ap.add_argument("--evidence", default=None, help="证据目录（manifest.json + evidence/run-*.log）")
    ap.add_argument("--current-main", default=None, help="当前 origin/main sha；用于检查基线是否落后")
    ap.add_argument("--require-current-main", action="store_true", help="基线不是当前 main 时直接判失败")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        with open(args.result, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"FAIL 读不了 bench 结果：{e}", file=sys.stderr)
        return 2

    checks, notes = [], []

    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if d.get("pr") != args.pr:
        check("pr", False, f"结果里的 pr={d.get('pr')} != {args.pr}")
    if d.get("sha") != args.sha:
        check("sha", False, f"结果里的 sha={str(d.get('sha'))[:12]} != {args.sha[:12]}")

    created = d.get("created_at", "")
    try:
        t = dt.datetime.fromisoformat(created.replace("Z", "+00:00"))
        age_h = (dt.datetime.now(dt.timezone.utc) - t.astimezone(dt.timezone.utc)).total_seconds() / 3600
        check("age", age_h <= args.max_age_hours, f"结果年龄 {age_h:.1f}h（上限 {args.max_age_hours}h）")
    except ValueError:
        check("age", False, f"created_at 无法解析：{created!r}")

    m = d.get("metrics") or {}
    dec = m.get("decode_tok_s") or {}
    a, b = dec.get("A") or [], dec.get("B") or []
    rounds_ok = len(a) >= args.min_rounds and len(b) >= args.min_rounds
    check("rounds", rounds_ok, f"decode 轮数 A={len(a)} B={len(b)}（下限 {args.min_rounds}）")

    gain = noise = req = 0.0
    if rounds_ok:
        gain = rel_gain(a, b)
        aa = m.get("aa_decode_tok_s") or []
        if len(aa) >= args.min_rounds:
            noise = (max(aa) - min(aa)) / median(aa)
        else:
            notes.append("未提供 A/A 噪声样本（aa_decode_tok_s），按 0 处理——建议补，否则 1% 级收益无法与噪声区分")
        req = max(args.min_gain, args.noise_mult * noise)
        check(
            "decode_gain",
            gain >= req,
            f"decode {median(a):.2f} → {median(b):.2f} tok/s，增益 {gain * 100:+.2f}%（要求 ≥ {req * 100:.2f}% = max({args.min_gain * 100:.1f}%, {args.noise_mult:g}×噪声 {noise * 100:.2f}%)）",
        )
        non_overlap = max(a) < min(b)
        check(
            "non_overlap",
            non_overlap,
            f"A∈[{min(a):.2f},{max(a):.2f}] B∈[{min(b):.2f},{max(b):.2f}]，组间{'不重叠' if non_overlap else '重叠'}",
        )
    else:
        check("decode_gain", False, "轮数不足，无法计算")

    pre = m.get("prefill_tok_s") or {}
    if pre.get("A") and pre.get("B"):
        pg = rel_gain(pre["A"], pre["B"])
        check(
            "prefill",
            pg >= -args.prefill_max_regression,
            f"prefill 增益 {pg * 100:+.2f}%（允许回退 {args.prefill_max_regression * 100:.1f}%）",
        )
    else:
        notes.append("没提供 prefill 数据，未做非回归检查")

    rss = m.get("rss_mb") or {}
    if rss.get("A") and rss.get("B"):
        rg = (rss["B"] - rss["A"]) / rss["A"]
        check(
            "rss",
            rg <= args.rss_max_regression,
            f"RSS {rss['A']} → {rss['B']} MB（{rg * 100:+.2f}%，允许 +{args.rss_max_regression * 100:.1f}%）",
        )

    cor = d.get("correctness") or {}
    check(
        "correctness",
        cor.get("llama_moe_verify_slot") is True,
        "LLAMA_MOE_VERIFY_SLOT byte 级自检" + ("通过" if cor.get("llama_moe_verify_slot") is True else "缺失/未通过"),
    )

    if args.evidence:
        ev = Path(args.evidence)
        man_file = ev / "manifest.json"
        man = None
        if not man_file.exists():
            check("evidence", False, f"缺少 {man_file}")
        else:
            try:
                man = json.loads(man_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                check("evidence", False, f"manifest 解析失败：{e}")
        if man is not None:
            check("evidence_pr_sha",
                  man.get("pr") == args.pr and man.get("sha") == args.sha,
                  "manifest 的 pr/sha 与评审对象一致")
            runs = man.get("runs") or []
            by_arm = {"A": 0, "B": 0}
            problems = []
            for r in runs:
                arm = r.get("arm")
                if arm in by_arm:
                    by_arm[arm] += 1
                lf = ev / (r.get("log") or "_missing_")
                if not lf.exists():
                    problems.append(f"缺日志 {r.get('log')}")
                    continue
                txt = lf.read_text(errors="replace")
                if "eval time" not in txt:
                    problems.append(f"{r.get('log')} 无 eval time")
                # llama.cpp 先打 "prompt eval time ... tokens per second"（prefill），
                # 再打 "eval time ... tokens per second"（decode）；只取后者。
                vals = []
                for line in txt.splitlines():
                    if "eval time" in line and "prompt eval time" not in line:
                        mm = re.search(r"([0-9.]+)\s*tokens per second", line)
                        if mm:
                            vals.append(float(mm.group(1)))
                if vals and r.get("decode_tok_s"):
                    got, want = vals[-1], float(r["decode_tok_s"])
                    if want and abs(got - want) / want > 0.02:
                        problems.append(f"{r.get('log')} 报 {want:.2f} 日志 {got:.2f}")
            runs_ok = (by_arm["A"] >= args.min_rounds and by_arm["B"] >= args.min_rounds and not problems)
            check("evidence_runs", runs_ok,
                  f"manifest 轮次 A={by_arm['A']} B={by_arm['B']}，日志核对{'通过' if not problems else '失败：' + '；'.join(problems[:3])}")
            dl = man.get("devlock") or {}
            check("evidence_devlock",
                  bool(dl.get("acquired_at")) and bool(dl.get("released_at")),
                  "devlock 取/放锁时间有记录")
            built = man.get("built") or {}
            check("evidence_binary",
                  isinstance(built.get("sha256"), str) and len(built["sha256"]) == 64,
                  "构建产物 sha256 已记录")
            if args.current_main:
                base_sha = (man.get("baseline") or {}).get("main_sha")
                if not base_sha:
                    if args.require_current_main:
                        check("baseline_current", False, "manifest 没记录 baseline.main_sha")
                elif base_sha == args.current_main:
                    check("baseline_current", True, f"基线就是当前 main（{base_sha[:8]}）")
                elif args.require_current_main:
                    check("baseline_current", False,
                          f"基线 main={base_sha[:8]} 已落后于当前 main={args.current_main[:8]}，需重跑")
                else:
                    notes.append(f"基线 main={base_sha[:8]} 落后于当前 main={args.current_main[:8]}（建议重跑；BENCH_REQUIRE_CURRENT_MAIN=1 可强制）")

    ok = all(c["ok"] for c in checks)
    out = {
        "pass": ok,
        "pr": args.pr,
        "sha": args.sha,
        "checks": checks,
        "notes": notes,
        "metrics": {
            "decode_gain": gain,
            "decode_noise": noise,
            "decode_required": req,
            "decode_median_a": median(a) if a else None,
            "decode_median_b": median(b) if b else None,
        },
    }
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(("PASS" if ok else "FAIL") + " bench 门禁")
        for c in checks:
            print(f"  [{'ok' if c['ok'] else 'XX'}] {c['name']}: {c['detail']}")
        for n in notes:
            print(f"  note: {n}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

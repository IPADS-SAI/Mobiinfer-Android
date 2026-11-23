#!/usr/bin/env bash
# bench 门禁的离线自测：不碰手机，只用合成数据验证 verify_bench.py / bench-gate.sh 的判据。
#
# 用法：scripts/pr-watch/test-bench-gate.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
PR=999
SHA=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef

mk() { # mk <out> <mode>
  python3 - "$1" "$2" "$PR" "$SHA" <<'PY'
import json, sys, datetime as dt
out, mode, pr, sha = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds")
d = {
    "pr": pr, "sha": sha, "created_at": now,
    "device": "vivo V2546A / SM8850 / HTP0 / v81",
    "command": "ssh vivo 'cd ~/csm/moe-exp && ./devlock ...'",
    "metrics": {
        "decode_tok_s": {"A": [15.10, 15.32, 15.21, 15.04], "B": [15.62, 15.51, 15.70, 15.44]},
        "prefill_tok_s": {"A": [21.0, 21.2, 20.9, 21.1], "B": [21.1, 21.0, 21.3, 20.8]},
        "aa_decode_tok_s": [15.12, 15.08, 15.20, 15.15],
        "rss_mb": {"A": 7000, "B": 7010},
    },
    "correctness": {"llama_moe_verify_slot": True},
}
if mode == "low_gain":
    d["metrics"]["decode_tok_s"]["B"] = [15.25, 15.20, 15.30, 15.18]
elif mode == "noisy":
    d["metrics"]["aa_decode_tok_s"] = [15.0, 15.9, 14.9, 16.0]
elif mode == "overlap":
    d["metrics"]["decode_tok_s"]["A"] = [15.10, 15.80, 15.21, 15.04]
elif mode == "stale":
    d["created_at"] = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=100)).isoformat()
elif mode == "no_correctness":
    d["correctness"]["llama_moe_verify_slot"] = False
elif mode == "wrong_pr":
    d["pr"] = pr + 1
json.dump(d, open(out, "w"), ensure_ascii=False, indent=1)
PY
}

expect() { # expect <file> <want_rc> <label>
  local rc=0
  python3 "$SCRIPT_DIR/verify_bench.py" "$1" --pr "$PR" --sha "$SHA" > "$TMP/out.txt" 2>&1 || rc=$?
  if [ "$rc" -eq "$2" ]; then
    echo "PASS  $3 (rc=$rc)"
  else
    echo "FAIL  $3：期望 rc=$2，实际 rc=$rc"; sed 's/^/      /' "$TMP/out.txt"; exit 1
  fi
}

mk "$TMP/good.json" good
mk "$TMP/low.json" low_gain
mk "$TMP/noisy.json" noisy
mk "$TMP/overlap.json" overlap
mk "$TMP/stale.json" stale
mk "$TMP/nocor.json" no_correctness
mk "$TMP/wrongpr.json" wrong_pr

expect "$TMP/good.json" 0 "收益+噪声+prefill+正确性都过"
expect "$TMP/low.json" 1 "1% 级收益被噪声/阈值挡住"
expect "$TMP/noisy.json" 1 "A/A 噪声过大时 +2.3% 也不认"
expect "$TMP/overlap.json" 1 "组间区间重叠"
expect "$TMP/stale.json" 1 "结果过期"
expect "$TMP/nocor.json" 1 "缺正确性自检"
expect "$TMP/wrongpr.json" 1 "PR 不匹配"

# bench-gate：缺文件
export PRWATCH_HOME="$TMP/state"
if "$SCRIPT_DIR/bench-gate.sh" 999 "$SHA" >/dev/null 2>&1; then
  echo "FAIL  bench-gate 缺文件时应失败"; exit 1
else
  echo "PASS  bench-gate 缺文件时拒绝"
fi

# bench-gate：BENCH_GATE=off 跳过
if BENCH_GATE=off "$SCRIPT_DIR/bench-gate.sh" 999 "$SHA" >/dev/null 2>&1; then
  echo "PASS  BENCH_GATE=off 时跳过"
else
  echo "FAIL  BENCH_GATE=off 应跳过"; exit 1
fi

# ---- 证据目录：manifest + 每轮日志 ----
mk_evidence() { # mk_evidence <dir> <mode>
  local ev=$1 mode=$2
  mkdir -p "$ev/evidence"
  python3 - "$ev" "$mode" "$PR" "$SHA" <<'PY2'
import json, sys
ev, mode, pr, sha = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
runs = []
A = [15.10, 15.32, 15.21, 15.04]
B = [15.62, 15.51, 15.70, 15.44]
for i, v in enumerate(A, 1):
    runs.append({"round": i, "arm": "A", "log": f"evidence/run-{i:03d}-A.log",
                 "decode_tok_s": v, "prefill_tok_s": 21.0, "eval_ms": 8470,
                 "ts": "2026-11-06T09:00:00+08:00"})
    open(f"{ev}/evidence/run-{i:03d}-A.log", "w").write(
        f"llama_print_timings: eval time = 8470.00 ms / 128 runs (66.17 ms per token, {v:.2f} tokens per second)\n")
for i, v in enumerate(B, 1):
    runs.append({"round": 10 + i, "arm": "B", "log": f"evidence/run-{10+i:03d}-B.log",
                 "decode_tok_s": v, "prefill_tok_s": 21.1, "eval_ms": 8200,
                 "ts": "2026-11-06T09:10:00+08:00"})
    open(f"{ev}/evidence/run-{10+i:03d}-B.log", "w").write(
        f"llama_print_timings: eval time = 8200.00 ms / 128 runs (64.06 ms per token, {v:.2f} tokens per second)\n")
if mode == "tamper":
    # 日志与 JSON 数字对不上
    open(f"{ev}/evidence/run-001-A.log", "w").write(
        "llama_print_timings: eval time = 9000.00 ms / 128 runs (70.31 ms per token, 14.20 tokens per second)\n")
man = {"pr": pr, "sha": sha,
       "built": {"llama_cli": "/tmp/llama-cli", "sha256": "0" * 64, "cmake_args": "cmake ..."},
       "model": {"path": "/data/models/x.gguf", "sha256": "1" * 64},
       "device": "vivo V2546A / SM8850 / HTP0 / v81",
       "devlock": {"acquired_at": "2026-11-06T08:58:00+08:00",
                   "released_at": "2026-11-06T09:20:00+08:00", "waited_s": 0},
       "runs": runs}
json.dump(man, open(f"{ev}/manifest.json", "w"), ensure_ascii=False, indent=1)
PY2
}

mk_evidence "$TMP/ev-good" good
mk_evidence "$TMP/ev-tamper" tamper

rc=0; python3 "$SCRIPT_DIR/verify_bench.py" "$TMP/good.json" --pr "$PR" --sha "$SHA" --evidence "$TMP/ev-good" >/dev/null 2>&1 || rc=$?
[ "$rc" -eq 0 ] && echo "PASS  证据完整时通过" || { echo "FAIL  证据完整应通过（rc=$rc）"; python3 "$SCRIPT_DIR/verify_bench.py" "$TMP/good.json" --pr "$PR" --sha "$SHA" --evidence "$TMP/ev-good"; exit 1; }

rc=0; python3 "$SCRIPT_DIR/verify_bench.py" "$TMP/good.json" --pr "$PR" --sha "$SHA" --evidence "$TMP/ev-tamper" >/dev/null 2>&1 || rc=$?
[ "$rc" -eq 1 ] && echo "PASS  日志与 JSON 不一致时拒绝" || { echo "FAIL  篡改日志应被拒（rc=$rc）"; exit 1; }

echo "全部通过"

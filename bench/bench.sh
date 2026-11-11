#!/data/data/com.termux/files/usr/bin/bash
# bench.sh — 一次 llama-cli 采样，参数全来自 env；用于参数臂矩阵与吞吐对比。
# 默认 = 2026-11-10 的生产基线：release-v81-sync（main + hexagon 11-10 后端同步）
# + O_DIRECT / H2 / Q4 / W80 / IO8。需要 profile 时显式 PROFILE=1。
#
# 关键 env（都有默认）：
#   REL, MODEL, EXPERTS, DRAFT, N, BANK, WINDOW, DEPTH, READ, IO_THREADS, PREFETCH,
#   HORIZON, QUEUE, MAX_ROWS, READY, REFRESH, H1_FRONT, PROBATION, IO_BUDGET, VERIFY_SLOT
# 输出：stdout 里由调用方解析 "[ Prompt: .. t/s | Generation: .. t/s ]"
#
# ⚠️ READ/DEPTH/WINDOW/PREFETCH/IO_THREADS 这类**只认短名**：直接传对应 LLAMA_MOE_*
#    会被这里的默认值覆盖（旧版行为，未改）。流水/新增开关（HORIZON/QUEUE/MAX_ROWS/
#    READY/REFRESH/H1_FRONT/…）短名优先，未给短名时回退读同名 LLAMA_MOE_*，
#    最后才是默认值（READY/REFRESH 默认 1，其余 0 / 8）。
set -u

ROOT=${ROOT:-$HOME/csm/moe-exp}
# 2026-11-10 起的生产基线构建。旧基线用 REL=$HOME/csm/moe/release-v81-pipe（11-08）。
REL=${REL:-$HOME/csm/moe/release-v81-sync}
MODEL=${MODEL:-$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/minicpm47-les-hexagon.gguf}
EXPERTS=${EXPERTS:-$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-htp-tiled}
DRAFT=${DRAFT:-$ROOT/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon/dspark-MiniCPM-V-4_7-q8_0.gguf}
N=${N:-64}
BANK=${BANK:-256}
# 窗口默认 80：实测最优（§3.22；80 与 96 打平，112 起撞内存墙）。回退用 WINDOW=52。
WINDOW=${WINDOW:-80}
DEPTH=${DEPTH:-8}
READ=${READ:-8}
# 8 = 2026-11-05 roofline 的 8 对 ABBA（+7.7%）；4 是 O_DIRECT 之前的旧默认。
IO_THREADS=${IO_THREADS:-8}
PREFETCH=${PREFETCH:-1}
# 跨层流水：提前几层提交 gate、队列里最多几个 job。默认值与 main 一致。
HORIZON=${HORIZON:-${LLAMA_MOE_LOOKAHEAD_HORIZON:-2}}
QUEUE=${QUEUE:-${LLAMA_MOE_PREFETCH_QUEUE:-4}}
# 预取/窗口覆盖的批宽。8 与 App 一致；4 是旧的 bench 值（SPECN>=4 时会把预取静默关掉）。
MAX_ROWS=${MAX_ROWS:-${LLAMA_MOE_PREFETCH_MAX_ROWS:-8}}
WINDOW_ROWS=${WINDOW_ROWS:-${LLAMA_MOE_WINDOW_MAX_ROWS:-8}}
# 默认 = "实测最快"配置（2026-11-10 同会话 ABBA：READY=1 → 21.05 t/s，
# +REFRESH=1 → 21.90 t/s，组间不重叠）。**生产默认仍是 READY=0/REFRESH=0**，
# 要测基线显式给 `READY=0 REFRESH=0`。
# 短名优先，其次长名（兼容旧脚本直接 export LLAMA_MOE_*），最后默认值。
# 空串按“没给”处理（否则 REFRESH='' 会被当成显式，却由 :- 重新取默认 1，
# 跳过下面的降级，最后在 init 里抛异常）。
_ref_explicit=0
[ -n "${REFRESH:-}" ] && _ref_explicit=1
[ -n "${LLAMA_MOE_PREFETCH_REFRESH:-}" ] && _ref_explicit=1
READY=${READY:-${LLAMA_MOE_EXPERT_READY:-1}}
REFRESH=${REFRESH:-${LLAMA_MOE_PREFETCH_REFRESH:-1}}
H1_FRONT=${H1_FRONT:-${LLAMA_MOE_PREFETCH_H1_FRONT:-0}}
PROBATION=${PROBATION:-${LLAMA_MOE_PF_PROBATION:-0}}
IO_BUDGET=${IO_BUDGET:-${LLAMA_MOE_EXPERT_IO_BUDGET:-0}}
VERIFY_SLOT=${VERIFY_SLOT:-${LLAMA_MOE_VERIFY_SLOT:-0}}
SPLIT=${SPLIT:-${LLAMA_MOE_MATMUL_SPLIT:-0}}
PROFILE=${PROFILE:-0}
FW_PROBE=${FW_PROBE:-0}
LV=${LV:-3}
PERF=${PERF:-0}
SPEC=${SPEC:-1}
SPECN=${SPECN:-3}
FA=${FA:-auto}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPT=${PROMPT:-"Explain how rain forms in clear simple English."}
SLOT_BANK_ARG=${BANK:-256}

# 显式要求 REFRESH=1 时**不**自动降级（让 init 尽早抛异常）；只有用默认值时才降级。
# 逐项对照 init_prefetch_refresh() / normalize_prefetch_depth() / 源码的取值语义：
#   READY=1、HORIZON>1、PREFETCH 首字符非 '0'（"00" 也算关）、READ atoi!=0
#   （负数=跟随 depth，不降级）、WINDOW>0 或 "auto"、DEPTH>0 且
#   BANK>=2*DEPTH+8（否则 depth 被归一化成 0）、H1_FRONT=0、
#   IO_BUDGET 数值=0（"00" 也是 0）、SPLIT 数值<=0（源码只有 >0 才禁用）。
_ref_reason=""
[ "$READY" = "1" ]        || _ref_reason="$_ref_reason READY!=1"
[ "$HORIZON" -ge 2 ]      || _ref_reason="$_ref_reason HORIZON<2"
case "$PREFETCH" in 0*) _ref_reason="$_ref_reason PREFETCH=0*" ;; esac
[ "$READ" -ne 0 ]         || _ref_reason="$_ref_reason READ=0"
case "$WINDOW" in
    auto) ;;
    *) [ "$WINDOW" -gt 0 ] || _ref_reason="$_ref_reason WINDOW<=0" ;;
esac
[ "$DEPTH" -gt 0 ]        || _ref_reason="$_ref_reason DEPTH<=0"
[ "$BANK" -ge $((2*DEPTH + 8)) ] || _ref_reason="$_ref_reason BANK<2*DEPTH+8"
[ "$H1_FRONT" != "1" ]    || _ref_reason="$_ref_reason H1_FRONT=1"
[ "$IO_BUDGET" -eq 0 ]    || _ref_reason="$_ref_reason IO_BUDGET!=0"
[ "$SPLIT" -le 0 ]        || _ref_reason="$_ref_reason SPLIT>0"
if [ "$REFRESH" = "1" ] && [ "$_ref_explicit" = "0" ] && [ -n "$_ref_reason" ]; then
    echo "[bench] REFRESH 默认 1 但不满足前置条件:$_ref_reason —— 自动降级为 0（要强制就显式 REFRESH=1）" >&2
    REFRESH=0
fi

export LD_LIBRARY_PATH="/system/lib64:/vendor/lib64:$REL/lib"
export ADSP_LIBRARY_PATH="$REL/lib;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp"
export GGML_HEXAGON_DEVICES=1
export GGML_HEXAGON_ARCH=81
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_MBUF=${MBUF:-256M,512M,4G}
export GGML_HEXAGON_OPPOLL=${OPPOLL:-1}
export GGML_HEXAGON_PROFILE=$PROFILE
export LLAMA_MOE_HTP_PACKED_DIR=$EXPERTS
export LLAMA_MOE_DECODE_WINDOW=$WINDOW
export LLAMA_MOE_WINDOW_MAX_ROWS=$WINDOW_ROWS
export LLAMA_MOE_IO_POOL=${IO_POOL:-1}
export LLAMA_MOE_IO_THREADS=$IO_THREADS
export LLAMA_MOE_LOOKAHEAD_PREFETCH=$PREFETCH
export LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH=$DEPTH
export LLAMA_MOE_LOOKAHEAD_HORIZON=$HORIZON
export LLAMA_MOE_PREFETCH_QUEUE=$QUEUE
export LLAMA_MOE_PREFETCH_MAX_ROWS=$MAX_ROWS
export LLAMA_MOE_PREFETCH_READ=$READ
export LLAMA_MOE_EXPERT_READY=$READY
export LLAMA_MOE_PREFETCH_REFRESH=$REFRESH
export LLAMA_MOE_PREFETCH_H1_FRONT=$H1_FRONT
export LLAMA_MOE_PF_PROBATION=$PROBATION
export LLAMA_MOE_EXPERT_IO_BUDGET=$IO_BUDGET
export LLAMA_MOE_VERIFY_SLOT=$VERIFY_SLOT
export LLAMA_MOE_MATMUL_SPLIT=$SPLIT
export LLAMA_MOE_READY_BATCH=1
READY_GROUP=${READY_GROUP:-2}
export LLAMA_MOE_READY_GROUP=$READY_GROUP
export LLAMA_MOE_READY_FAST_FIRST=0
export LLAMA_MOE_READY_TRACE=${READY_TRACE:-0}
export LLAMA_MOE_SKIP_READ=${SKIP_READ:-0}
export LLAMA_MOE_SWEEP=${SWEEP_PATH:-}
export LLAMA_MOE_DRAFT_THRESHOLD=${DRAFT_THRESHOLD:-}
export LLAMA_MOE_DUMMY_READ=${DUMMY_READ:-0}
export LLAMA_FW_PROBE=$FW_PROBE

if [ -n "${PROMPT_FILE:-}" ]; then PROMPT_ARGS=(-f "$PROMPT_FILE"); else PROMPT_ARGS=(-p "$PROMPT"); fi
# EXTRA：透传任意额外 llama-cli 参数（例如 EXTRA="-no-cnv"）。
# 注意：本脚本只认**短名** env（PREFETCH/SKIP_READ/DUMMY_READ/READ/DEPTH/
# IO_THREADS…），传 LLAMA_MOE_* 会被这里 export 的默认值覆盖。
EXTRA=${EXTRA:-}

LIB_SHA=$(sha256sum "$REL/lib/libllama.so" 2>/dev/null | cut -c1-12)
echo "[bench] rel=$REL lib=${LIB_SHA:-NA} n=$N window=$WINDOW depth=$DEPTH read=$READ io=$IO_THREADS spec=$SPECN horizon=$HORIZON queue=$QUEUE maxrows=$MAX_ROWS ready=$READY refresh=$REFRESH h1front=$H1_FRONT probation=$PROBATION iobudget=$IO_BUDGET verify=$VERIFY_SLOT split=$SPLIT profile=$PROFILE" >&2

exec "$REL/bin/llama-cli" -m "$MODEL" \
  --moe-experts "$EXPERTS" --moe-slot-bank "$BANK" \
  --device HTP0 -ngl 99 \
  ${NOMAP:+--no-mmap} -c ${CTX:-2048} -b ${BATCH:-512} -ub ${UB:-128} -ctk ${CTK:-f16} -ctv ${CTV:-f16} -fa "$FA" -fit off --no-warmup \
  --seed 42 --temp 0 --single-turn --reasoning off -lv "$LV" \
  "${PROMPT_ARGS[@]}" $EXTRA \
  ${SPEC:+$([ "$SPEC" = "1" ] && echo "--spec-type draft-dspark --model-draft $DRAFT --device-draft HTP0 -ngld 99 --spec-draft-n-max $SPECN ${SPECN_MIN:+--spec-draft-n-min $SPECN_MIN} ${SPECPMIN:+--spec-draft-p-min $SPECPMIN}")} -n "$N" ${PERF:+$([ "$PERF" = "1" ] && echo --perf)}

#!/usr/bin/env bash
# 生成评审沙箱的 bwrap 参数。**生产（review-one.sh）与离线测试（test-packet.sh）共用**
# 这一份构造逻辑，否则测试自己手写一遍 bwrap，生产把 `-C` 退回 /tmp/work 也照样 PASS
# ——那正是 2026-11-10 那次"材料包在沙箱里读不到"事故的形态。
#
# 用法：
#   source sandbox-args.sh
#   prwatch_sandbox_args <wt> <codex_cwd> <outdir> <codex_home> <real_home> <script_dir> <git_common>
#   bwrap "${PRWATCH_ARGS[@]}" <命令...>
#
# 产出：
#   PRWATCH_ARGS       bwrap 的参数数组（不含 bwrap 本身、不含要执行的命令）
#   PRWATCH_BWRAP_CWD  沙箱内的工作目录（材料包模式是 /tmp/packet，全量模式是 /tmp/work）
#
# 隔离约定：--ro-bind / / 打底，/home 换成 tmpfs（拿不到 ~/.ssh、~/.config/gh），
# 只显式挂 worktree 或材料包、只读的脚本目录、codex 家目录、输出目录，以及
# $REAL_HOME/.local（PATH 里的工具，原行为）。写权限只落在 /tmp/out、/tmp/codex、/tmp。
prwatch_sandbox_args() {
  local WT=$1 CODEX_CWD=$2 OUT=$3 CODEX_HOME=$4 REAL_HOME=$5 SCRIPTS=$6 GIT_COMMON=$7
  PRWATCH_ARGS=(--die-with-parent
    --unshare-pid --unshare-ipc --unshare-uts --unshare-cgroup
    --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp
    --bind "$OUT/codex-out" /tmp/out
    --ro-bind "$SCRIPTS" /tmp/scripts
    --bind "$CODEX_HOME" /tmp/codex
    --ro-bind "$REAL_HOME/.local" /tmp/local)
  PRWATCH_BWRAP_CWD=/tmp/work
  if [ "$CODEX_CWD" != "$WT" ]; then
    # 材料包模式：包目录就是工作目录，packet.md 里的 files/<path>、AGENTS.md 才找得到。
    # 反过来（旧事故）：这里退回 /tmp/work，模型读不到材料包还得去猜路径。
    PRWATCH_ARGS+=(--ro-bind "$CODEX_CWD" /tmp/packet)
    PRWATCH_BWRAP_CWD=/tmp/packet
  else
    PRWATCH_ARGS+=(--bind "$WT" /tmp/work)
    # /home 被 tmpfs 遮住，但 git 需要 worktree 的 .git 指针（指向真实 gitdir），
    # 所以把 common gitdir 只读绑进来，并在原路径建软链让指针能解析。
    if [ -e "$CODEX_CWD/.git" ] && [ -d "$GIT_COMMON" ]; then
      PRWATCH_ARGS+=(--ro-bind "$GIT_COMMON" /tmp/gitdir)
    fi
  fi
  PRWATCH_ARGS+=(--tmpfs /home --dir "$REAL_HOME")
  local acc=/home part rel=${GIT_COMMON#/home/}
  if [ "$PRWATCH_BWRAP_CWD" = /tmp/work ] && [ "$rel" != "$GIT_COMMON" ] && [ -d "$GIT_COMMON" ]; then
    local IFS=/
    for part in $rel; do
      case "$part" in .git) continue;; esac
      acc="$acc/$part"
      PRWATCH_ARGS+=(--dir "$acc")
    done
    PRWATCH_ARGS+=(--symlink /tmp/gitdir "$GIT_COMMON")
  fi
  PRWATCH_ARGS+=(--symlink /tmp/codex "$REAL_HOME/.codex"
    --symlink /tmp/local "$REAL_HOME/.local"
    --chdir "$PRWATCH_BWRAP_CWD"
    --setenv HOME "$REAL_HOME" --setenv TMPDIR /tmp
    --setenv PATH "/tmp/local/bin:/usr/local/bin:/usr/bin:/bin")
}

# 生成 `codex exec` 的参数（不含 bwrap 外壳、timeout 与末尾的 `-`）。
# 生产 run_codex 与离线测试共用，这样"生产把 -C 或启动参数改错"测试也能抓到。
#
# 用法：prwatch_codex_args <cwd> <model> <schema> <verdict-out>
# 产出：PRWATCH_CODEX_ARGS
#
# --skip-git-repo-check：材料包模式的工作目录不是 git 仓库（/tmp/packet）。本机 codex
# 目前靠 wrapper 强制 --dangerously-bypass-approvals-and-sandbox 才能跑起来，实测不传
# 这个参数也能启动；但不该依赖 wrapper 的隐式行为，显式传更稳（2026-11-10 定）。
prwatch_codex_args() {
  local cwd=$1 model=$2 schema=$3 out=$4
  PRWATCH_CODEX_ARGS=(exec
    -C "$cwd"
    -c approval_policy=never
    -m "$model"
    --skip-git-repo-check
    --output-schema "$schema"
    -o "$out"
    --json)
}

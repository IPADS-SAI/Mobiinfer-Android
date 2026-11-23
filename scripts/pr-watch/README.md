# pr-watch：PR 自动评审 + 自动合并

轮询本仓库的 open PR，对新 head 跑 **独立 review agent**（默认 `gpt-6.1-sol`），
贴出「阻塞项 / 非阻塞项 / 结论」，并按 `MERGE_POLICY` 决定是否自动 squash 合并。

目标是让「分支 → PR → 独立 review → 合并」这条 AGENTS.md §2 的流程**机械化、可审计**，
而不是让某个 agent 在对话里凭印象说"可以合"。

## 快速开始

```sh
# 1) 起守护进程（脱离终端，默认 120s 轮询）
scripts/pr-watch/start.sh

# 2) 看状态
scripts/pr-watch/status.sh

# 3) 只跑一轮（调试用，不进入循环）
scripts/pr-watch/watch.sh --once

# 4) 只审某个 PR，不发评论/不合并
DRY_RUN=1 scripts/pr-watch/review-one.sh 4

# 5) 停
scripts/pr-watch/stop.sh
```

## 一次评审做了什么

1. `gh pr view` 取元信息；draft / fork / `hold` / `do-not-merge` 直接跳过。
2. 把 PR head 拉进独立 worktree（默认 `~/pr-review/pr-<N>`），不碰你正在改的工作区。
3. **T0 静态门禁**（bash 判，不依赖模型）：
   - `artifacts/`、`*.gguf`、`*.jsonl`、`*.stderr/stdout`、`node_modules/`、`profile-data.json` 进了 git → FAIL
   - 单文件 > 5 MB → FAIL
   - 疑似密钥（`ghp_`、`sk-`、`AKIA`、`PRIVATE KEY`）→ FAIL
   - 改动 `exps/*/README.md` 但缺「实验内容 / 复现命令 / 结果」→ FAIL
   - 改 `.github/workflows/**` → FAIL（当前 token 无 `workflow` scope）
4. `codex exec` 在 worktree 里做 review，强制 JSON schema 输出；T1（可选）在 **bwrap 沙箱**里
   `cmake` 构建 + `ctest`，无网络、`/home` 被 tmpfs 遮住，只绑 worktree。
5. 把结论贴成 PR 评论，按结论打 `review:approve` / `review:changes` / `review:blocked` / `review:error`；
   同时落盘 `gate.json`（T0/T1/结论/policy_violations，绑定 head）。
6. `merge-ready.sh` 只信 `gate.json` + 实时 PR 状态：approve、T0 过、T1 未失败、
   policy_violations 为空、head 未变、非 draft/fork、目标 `main`、无 `hold` →
   `gh pr merge --squash --delete-branch --match-head-commit`。
7. 已 approve 但门禁暂未过（例如 `merge-ok` 还没打、`mergeable=UNKNOWN`）→ 进
   `pending-merge.txt`，之后**只重试门禁**，不重跑评审；head 变了或 PR 关了才出队。
8. 全过程写 `audit.jsonl`。

## 目录与状态

| 路径 | 内容 |
|---|---|
| `scripts/pr-watch/` | 脚本、prompt、schema（进 git） |
| `~/.local/state/pr-watch/` | `watch.log`、`audit.jsonl`、`seen.txt`、`rate.txt`、`pending-merge.txt`、`pr-<N>-<sha>/`、`bench/pr-<N>-<sha>.json`（不进 git） |
| `~/pr-review/pr-<N>` | 每次评审的临时 worktree（不进 git） |
| `~/.codex/` | codex 家目录（模型 catalog / 认证） |

可用 `PRWATCH_HOME` / `WT_DIR` / `CODEX_HOME` 覆盖；在被沙箱限制的环境里把它们指到可写目录。

## bench 门禁（改到 src/ggml/tools 就必须有证据）

`BENCH_GATE=auto`（默认）时，PR 只要改到 `src/ ggml/ tools/ common/ examples/`，
合并前必须有一份**按 head 绑定**的 bench 结果，且通过统计判据：

```sh
# 跑完 skills/bench 的 A/B（ABBA、3+ 轮、devlock 串行）后，把它记成 JSON 并安装
scripts/pr-watch/bench-record.sh 8 <head-sha> bench-result.json
# → ~/.local/state/pr-watch/bench/pr-8-<sha>.json，并立即校验
```

判据（`verify_bench.py`，全部可配）：

| 检查 | 默认阈值 |
|---|---|
| decode 轮数 | A、B 各 ≥ 4 轮 |
| decode 增益 | ≥ max(1.5%, A/A 噪声)，且 A、B 的 min–max **区间不重叠** |
| prefill | 回退 ≤ 2% |
| RSS | 增加 ≤ 2% |
| 正确性 | `correctness.llama_moe_verify_slot == true`（byte 级自检） |
| 新鲜度 | 结果 ≤ 72h，且 `pr`/`sha` 与当前评审对象一致 |

阈值在 `config.env`：`BENCH_MIN_GAIN` / `BENCH_NOISE_MULT` / `BENCH_MIN_ROUNDS` /
`BENCH_PREFILL_MAX_REGRESSION` / `BENCH_RSS_MAX_REGRESSION` / `BENCH_MAX_AGE_HOURS`。
`BENCH_GATE=required` 一律要求，`off` 关闭。

格式见 `bench-result.schema.json` 与 `bench-result.example.json`；离线自测：

```sh
scripts/pr-watch/test-bench-gate.sh     # 合成数据，不碰手机
```

## 复现 agent（Stage 1，默认 DeepSeek）

便宜模型只做**复现**，不写结论；判决仍归 GPT（`REVIEW_MODEL`）。

```sh
# 先看前置条件（设备锁 / SDK / NDK / codex），不花模型额度
scripts/pr-watch/bench-agent.sh 8 <head-sha> --check

# 设备被占时排队（默认 0 = 立刻退出，不占着设备空等）
scripts/pr-watch/bench-agent.sh 8 <head-sha> --wait-lock 3600

# 换模型（默认 group/wb-deepseek）
scripts/pr-watch/bench-agent.sh 8 <head-sha> --model group/wb-deepseek
```

做的事：建独立 worktree → 让 agent 读 `skills/bench/SKILL.md` → 构建 v81 →
`ssh vivo` + `devlock` → ABBA ≥4 轮 → 产出 `bench-result.json`、`manifest.json`、
`evidence/run-*.log` → 宿主跑 `verify_bench.py --evidence` 校验 → 通过才
`bench-record.sh` 安装到 `~/.local/state/pr-watch/bench/pr-<N>-<sha>.json`。

证据校验（防"模型说自己跑了"）：

| 检查 | 内容 |
|---|---|
| manifest 绑定 | `pr`/`sha` 与评审对象一致 |
| 轮次 | manifest 里 A、B 各 ≥4 轮，且每轮日志文件存在 |
| 日志自洽 | 日志里 `tokens per second` 与 `bench-result.json` 相差 ≤2% |
| 独占设备 | `devlock.acquired_at/released_at` 有记录 |
| 构建来源 | `built.sha256` 已记录（64 hex） |

退出码：`0` 成功；`2` 前置条件不满足；`3` 设备被占；`4` 复现失败；`5` 校验不通过。

**安全边界**：这个 agent 需要在本机构建、`ssh vivo`、跑 `devlock`，所以**不能用 bwrap 隔离**，
只能在你自己发起的场景下跑，且只对`本仓库`的 PR 使用；prompt 里禁止它改仓库、push、写结论。

要自动化（watch 发现缺证据就自动起复现）把 `BENCH_AUTO_REPRO=1`，默认关闭——它可能长时间占用设备。

## 材料包（Stage 3）：贵模型只看证据 + 高风险 diff

`REVIEW_PACKET=1`（默认）时，GPT 不再自己 `git diff` 全仓，而是拿到一个裁好的包：

| 文件 | 内容 |
|---|---|
| `packet/packet.md` | PR 元信息 + T0 + bench 门禁状态 + 低风险摘要 + 高风险 diff（有上限） |
| `packet/files/<path>` | 高风险文件全文（标 `full` 的才有；标 `omitted` 的没有） |
| `packet/high.diff` | 高风险文件完整 diff（落盘保留全文；进入 `packet.md` 的部分超 `PACKET_MAX_DIFF_BYTES` 时截断并留痕） |
| `packet/low.summary.md` | 逐文件改动统计 + 便宜模型（`group/wb-deepseek`）的摘要 |
| `packet/stats.json` | 规模统计（写进评论，便于追踪成本） |

**材料包目录就是评审沙箱的工作目录**：`packet.md` 里一律用包内相对路径
（`files/<path>`、`AGENTS.md`）。这条以前是错的——旧版把工作目录写死成 worktree，
模型按 `packet/files/...` 去读全部落空，于是每次都多出一条"材料缺失"阻塞项。

风险分类看**改动会不会影响行为/测量**，不看顶层目录：`*.c/*.cpp/*.h/*.sh/*.py`、
`CMakeLists.txt`、`src/ ggml/ tools/ common/ examples/ include/ tests/ app/ ci/ cmake/
bench/ scripts/` 都算高风险；`*.md/*.txt`/图片等散文与数据算低风险。

低风险摘要**不能**当结论用：`prompt-packet.md` 明确要求 GPT 只对高风险文件逐行审，
低风险只做政策检查。摘要只是观点，**逐文件 +/- 统计是脚本写的**，不会因为摘要跑偏而丢。

### 材料不够时：自动补一轮（不是直接 block）

verdict 的 `needs_full_diff` 里点名、且确实属于本 PR 改动的文件，编排器会自动用
`PACKET_EXTRA` 重开一轮评审（`REVIEW_AUTO_EXPAND=1`，最多 `PACKET_MAX_ATTEMPTS=2` 轮），
并带上上一轮已给全文的文件（`stats.json.full_files`），避免它们被新点名的大文件挤出预算。
补材料轮用更宽的 `PACKET_EXTRA_MAX_FILES_BYTES`（默认 600 KB）。

人工升级仍然可以指定文件：

```sh
PACKET_EXTRA="exps/foo/README.md docs/bar.md" scripts/pr-watch/review-one.sh 8 <sha>
```

口径约定（`prompt-packet.md`）：

- bench 门禁 `不适用`（PR 没碰运行时路径）时，**缺 bench 结果不是阻塞理由**；
- T1 未运行是 WARN，只有 T1 明确 FAIL 才阻塞；
- 材料被裁剪本身不是缺陷——先看 diff，diff 够就出结论。

### 补材料轮用尽仍缺材料 → fail closed

跑满 `PACKET_MAX_ATTEMPTS` 后，如果模型仍然点名要本 PR 改过的文件的全文（说明它没看到
自己想看的东西），此时的 `approve` **不算数**：`fail-closed.sh` 会把结论降级为 `blocked`
并在阻塞项里写明缺哪些文件。理由很直接——让 `approve` 过门禁等于给"没审到的那部分"签字，
而门禁是会自动合并的。人工确认后可以用 `PACKET_EXTRA` 补全再审。

### 目录里几个小工具（都能单独测）

| 脚本 | 作用 | 测试 |
|---|---|---|
| `review-packet.sh` | 生成材料包（高风险 diff/全文、低风险统计、bench 门禁状态） | `test-packet.sh` |
| `pick-paths.sh` | 从 `needs_full_diff` 自由文本里挑出**本 PR 真改过**的路径 | `test-packet.sh` |
| `sandbox-args.sh` | 生成评审沙箱的 bwrap 参数（生产与测试**共用**，防 `-C` 退化） | `test-packet.sh` |
| `fail-closed.sh` | 补轮用尽仍缺材料时把 `approve` 降级为 `blocked` | `test-packet.sh` |

关掉退回全量模式：`REVIEW_PACKET=0`。相关阈值：`PACKET_MAX_DIFF_BYTES` /
`PACKET_MAX_FILES_BYTES` / `PACKET_EXTRA_MAX_FILES_BYTES` / `PACKET_LOW_DIFF_BYTES` /
`PACKET_SUMMARY_MODEL` / `REVIEW_AUTO_EXPAND` / `PACKET_MAX_ATTEMPTS`。

离线自测（不联网、不叫模型；覆盖 PR #16 / #14 两类事故）：

```sh
scripts/pr-watch/test-packet.sh
```

## 标签

| 标签 | 含义 |
|---|---|
| `review:approve` / `review:changes` / `review:blocked` / `review:error` | bot 自动维护的结论 |
| `hold` / `do-not-merge` | 人工刹车：跳过评审与合并（`HOLD_LABELS`） |
| `review:again` | 强制重跑一次（同 head 也会重审，跑完自动摘掉） |
| `run-build` | 本 PR 额外跑 T1 构建（`cmake` + 单测） |
| `merge-ok` | `MERGE_POLICY=label` 时的人工放行标签 |

## 合并策略

`config.env` 的 `MERGE_POLICY`：

- `auto`（默认）：评审 approve 即合并。**这是对 AGENTS.md §2「由人类执行合并」的例外**，
  授权与边界见 `policy.md`。
- `label`：需要人在 PR 上打 `merge-ok` 才合并。
- `report`：只审不合。

改策略改 `config.env` 后重启：`stop.sh && start.sh`。

## 成本与失控保护

- `MAX_REVIEWS_PER_HOUR`（默认 4）硬上限；同一 head SHA 只审一次。
- 单次 `REVIEW_TIMEOUT`（默认 1800s）；T1 每段另有 `BUILD_TIMEOUT`，worker 总预算
  会按 `RUN_BUILD_CHECK` 把两段构建算进去，避免内外层超时打架。
- 结论拿不到（超时/报错）→ `review:error`，**永不合并**。
- 基础设施失败（gh 网络、worktree 冲突等）不占每小时配额；同一 head 连续失败 3 次
  就暂停该 PR（head 变化自动恢复），避免 120s 一次地空转。
- 已 approve 未合并的进 `pending-merge.txt`，只重试门禁（几秒），不会再调模型。

## 安全

- codex 进程的 `GH_TOKEN` / `GITHUB_TOKEN` 被显式 unset；PR 内容按不可信输入处理（prompt 里写死）。
- 本机 `/home/yydh/.local/bin/codex` 是 wrapper，**强制 `--dangerously-bypass-approvals-and-sandbox`**，
  所以 `-s workspace-write` 实际不生效。真正的隔离在外层 bwrap：
  `--ro-bind / /` + `/home` 换 tmpfs（`~/.ssh`、`~/.config/gh`、`~/.gitconfig` 全部不可见），
  只挂 worktree、输出目录、只读的 scripts、`CODEX_HOME` 和 `~/.local`（codex 自身），
  保留网络供模型 API。没有 bwrap 时退化为空 HOME（弱隔离），评论里会注明。
- T1 构建另用一套 bwrap：`--unshare-all` **断网**，`/home`/`/tmp` 空 tmpfs，只把 worktree
  绑到 `/tmp/work`。没有 bwrap 时默认**拒绝构建**，除非显式
  `ALLOW_UNSANDBOXED_BUILD=1`（等于接受在宿主跑未审代码）。
- 评审前后都校验 worktree 仍等于被审 head（`HEAD` 未变、`status --porcelain` 干净），
  否则结论作废、拒绝合并。
- fork PR 也评审；`ALLOW_FORK_MERGE=true` 时**允许自动合并**（2026-11-06 人类要求），但仍要求结论 approve、T0 全过、head 未变。fork 的 diff 是不可信输入，风险由仓库 owner 承担。
- 只审目标为 `main` 的 PR；`gate.json` 同时绑定 `head` 与 `base`，base 变了不会被复用。
- bot 只合并目标为 `main`、非 draft、head 未变、T0 通过、无 policy_violations 的 PR；fork 需要 `ALLOW_FORK_MERGE=true`。

## 排障

- `codex exec` 报 `Read-only file system`：`CODEX_HOME` 不可写，指到可写目录或换非沙箱 shell。
- 守护进程起不来：看 `~/.local/state/pr-watch/daemon.log`；`flock` 探测是否已有实例。
- `stop.sh` 杀的是 **session**（setsid 的 sid == pid）并补上进程组：GNU `timeout` 会给
  子进程新建进程组但不换 session，只杀组会漏掉它。仍存活会打印警告并以 1 退出。
- 想验证不会误合并：`DRY_RUN=1` 跑单 PR，检查 `comment.md`、`verdict.json`、`gate.json`。
- `gh pr review --approve` 不可用：仓库只有你一个账号，GitHub 不允许自己批自己；
  所以 bot 用「评论 + label」表达结论。要真正的绿色 Approve 需装一个 GitHub App。

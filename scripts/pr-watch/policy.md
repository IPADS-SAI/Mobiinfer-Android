# pr-watch 授权边界（AGENTS.md §2 的机械化实施）

## 身份

- bot 由 `scripts/pr-watch/` 驱动，以仓库 owner 的 `gh` 凭证执行。
- 它**不是**写 PR 的那个 agent：它只读 diff、只评审、按规则合并，不修改 PR 分支。

## 可以做什么

1. 对 open PR 跑独立评审，输出「阻塞项 / 非阻塞项 / 政策检查 / 结论」。
2. 贴评论、打/摘 `review:*` 标签。
3. 在满足**全部**条件时 squash 合并到 `main`：
   - 结论 `approve`，`blocking` 为空，无 `policy_violations`；
   - T0 静态门禁全过（数据文件、大文件、密钥、exps 文档三块、workflow）；
   - T1 构建未失败（未跑不算失败）；
   - bench 证据达标（改到 `src/`/`ggml/`/`tools/`/`common/`/`examples/` 时）：
     `bench-result.json` 通过 `verify_bench.py`（≥4 轮、增益 ≥ max(1.5%, A/A 噪声)、
     组间不重叠、prefill/RSS 不回退、正确性自检通过、≤72h 且绑定当前 head）；
   - PR 非 draft、目标 `main`、`mergeable`，且 head 与评审时一致（fork PR 需要 `ALLOW_FORK_MERGE=true`）；
   - 无 `hold` / `do-not-merge` 标签；
   - `MERGE_POLICY=auto`（或 `label` 模式下已有人打 `merge-ok`）。
4. 合并后写审计记录，并在 PR 上回帖说明合并依据。

## 不可以做什么

- 不修改 PR 代码、不 push 到任何分支、不 rebase、不 force-push。
- 不合并自己 `request_changes` / `blocked` / `error` 的 PR。
- 不跳过 `hold` 标签，不合并 fork PR，不合并改 `.github/workflows/**` 的 PR。
- 不在没有 T0 通过的情况下合并文档类 PR 之外的任何东西。

## 人工干预入口

| 动作 | 方式 |
|---|---|
| 暂停某个 PR | 打 `hold` 标签 |
| 全局停摆 | `scripts/pr-watch/stop.sh` |
| 改回人工合并 | `MERGE_POLICY=label` 或 `report`，重启 |
| 重跑评审 | 打 `review:again` 标签 |
| 审计追溯 | `~/.local/state/pr-watch/audit.jsonl` |

## 变更记录

- 2026-11-05：人类在当轮对话明确要求「PR 用 codex 自动验证并合并」，并指定评审模型
  `gpt-6.1-sol`。据此在 AGENTS.md §2 增加「自动评审 bot」条款，把「由人类执行合并」
  改为「由独立 review bot 在门禁通过后自动合并；人类保留 `hold` 刹车」。
  本目录即该条款的实现与证据。

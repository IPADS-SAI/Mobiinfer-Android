# hexagon-ui-fixes 合并前审查

审查对象：`/home/yydh/llama.cpp-omni-fixes` 相对 `081f6f3` 的未提交改动。目标分支为 `hexagon-minicpm-dspark`，当前已保存提交 `483e503`。源 worktree 未修改、未提交。

## 需要先修复的问题

1. **高：DFlash 被当作 DSpark 执行。** 源 worktree 的 `minicpm47_runtime.cpp:142–153` 同时自动发现 DFlash 和 DSpark 文件，但无条件选择 `COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK`。新增 Qwen3.5 配置明确使用 DFlash。`common/speculative.cpp` 根据类型选择不同的 block token 数与采样位置，并非等价名称。应识别草稿类型，分别设置 DFlash/DSpark，不能只识别文件路径。

2. **中：视觉历史恢复缺少 assistant 回合边界。** 源 worktree 的 `llama_jni.cpp:1149–1151` 在视觉路径只回放 assistant 正文；下一条 user 在 `1126` 直接追加 `<|im_start|>user`，中间没有 `<|im_end|>`。恢复多轮视觉历史后输入的对话序列不完整，可能影响续聊。应按模型聊天模板还原完整历史，并避免把固定 ChatML 格式当作通用视觉格式。

## 合并准备

- 完整源改动（含新增资源）已快照到 `/tmp/hexagon-ui-fixes-review/worktree.patch`。
- 对当前分支执行普通 `git apply --check` 确认存在冲突，主要涉及 JNI、运行时、会话存储、主界面和字符串；不能直接覆盖。
- 解决冲突时须保留当前 batch/ubatch=512、Python 高亮、首页示例、DSP 等待设置、生命周期保护，以及默认加载 mmproj 的最新要求。
- 审查为静态代码检查；未构建或真机验证源 worktree，尚未合并。

建议先修复上述问题，再手工整合重叠文件，验证多轮历史恢复、模型切换及加载失败后的恢复。

## 选择性整合与验证

按用户要求，DFlash、Qwen3.5-MoE、多轮对话和图片历史恢复全部排除，源 worktree 保持原状。当前分支只整合模型标题、模型选择菜单与选中图标、模型管理页运行标识，以及 JNI 空状态检查和 reset 失败状态处理。加载失败、卸载和销毁时清除运行模型标识。下载 ETA 的重试统计尚需进一步检查，未合入。

首页两个 suggestions（显示文字及实际问题）、ChatAdapter、ModelInfo 和 ConversationStore 均与 `483e503` 一致。默认 b512/ub512、bank256、cache48+prefetch4、8GB 预算；mmproj 默认随模型加载。

验证：`./gradlew :app:assembleDebug --offline` 成功（4分36秒），debug APK 已安装到一加13。UI 自动检查通过模型标题、模型下拉选中标识、模型管理页；确认没有 Qwen3.5 条目。

App 单轮冒烟测试使用中文 Python 斐波那契问题，41 输入/128 输出，prefill **22.64 tok/s**、decode **17.59 tok/s**，草稿接受 91/105，峰值 RSS **952.04 MiB**。日志确认 `mmproj: loading`；输出为 Python 代码块，受128-token测试上限截断，不代表完整函数正确性验证。未测试图片输入和下载网络流程；未据此声称整体性能提升。

原始结果前缀：`app-ui-safe-mmproj-r1`，位于 `/home/yydh/moe-speculative/hexagon-results/oneplus-app-20261017/`。UI XML 和截图位于 `/home/yydh/moe-speculative/hexagon-results/app-ui-safe-20261017/`，构建日志 `/tmp/app-ui-safe-build.log`。本次整合随后按用户要求提交，versionCode 设为 1，versionName 保持 0.1。

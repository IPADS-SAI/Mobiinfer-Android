#!/usr/bin/env python3
"""从 gguf 里剥掉 MoE 专家张量，产出一个「只含常驻权重」的小 gguf。

为什么需要它
------------
MoE-SSD 下专家的内容来自 sidecar（运行时 preadv 写入），gguf 里那份**从头到尾
没被读过** —— `virtualize_expert_tensor` 会把它们从 weights_map 里摘掉。所以
gguf 里那份专家数据是纯死重量（V47 上是 18.7 GiB 里的 ~17.6 GiB）。

它还有个副作用：加载器启用 mmap 时会把**整个文件**包成一个 Metal buffer
（因为 tensor 的 file offset 从文件头散到尾），于是映射区 ≈ 18.7 GiB，
远超设备 RAM，prefill 直接 OOM。

剥掉之后 gguf 只剩 ~1.1 GiB：
  - mmap 的映射区变成 ~1.1 GiB，可以正常工作
  - 常驻权重变成「可回收的文件页」，不再和 page cache 抢匿名内存
  - 少占 ~17.6 GiB 磁盘

专家的形状/类型由 sidecar manifest 提供，所以剥掉之后引擎仍然能建出
（空内容、待 preadv 填充的）专家张量。

用法
----
  python strip_experts.py --model in.gguf --sidecar <sidecar 目录> --out out-noshard.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# tools/moe/strip_experts.py -> parents[2] = 仓库根（再拼 gguf-py）。脚本挪位置要同步改深度。
LLAMA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))

from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402


def expert_names(sidecar_dir: Path) -> set[str]:
    """从 sidecar manifest 取专家的 gguf 张量名。

    用 manifest 而不是名字模式匹配：manifest 是「哪些张量被搬进 sidecar」的权威记录，
    而名字模式（比如以 `_exps` 结尾）会随架构变化、也可能误伤。
    """
    manifest = sidecar_dir / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"sidecar manifest 不存在: {manifest}")
    entries = json.loads(manifest.read_text())["entries"]
    names = {e["original_gguf_tensor_name"] for e in entries}
    if not names:
        raise SystemExit("manifest 里没有任何 entry")
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description="剥掉 gguf 里的 MoE 专家张量")
    ap.add_argument("--model", required=True, help="源 gguf")
    ap.add_argument("--sidecar", required=True, help="sidecar 目录（读 manifest.json）")
    ap.add_argument("--out", required=True, help="输出 gguf")
    ap.add_argument("--keep-experts", action="store_true",
                    help="只做一次「读进来再写出去」的往返，用于验证工具本身不改变模型")
    args = ap.parse_args()

    src = Path(args.model)
    out = Path(args.out)
    if not src.exists():
        raise SystemExit(f"源 gguf 不存在: {src}")

    drop = set() if args.keep_experts else expert_names(Path(args.sidecar))
    if not args.keep_experts:
        print(f"将从 manifest 剥掉的专家张量: {len(drop)} 个")

    print(f"读取 {src} ({src.stat().st_size / 2**30:.2f} GiB) ...")
    reader = GGUFReader(src)

    arch_field = reader.fields.get("general.architecture")
    if arch_field is None:
        raise SystemExit("源 gguf 缺 general.architecture")
    arch = arch_field.contents()

    writer = GGUFWriter(out, arch, use_temp_file=True)

    # 1) KV：逐个搬运
    #    - `general.architecture`：writer 构造函数已写，跳过避免重复
    #    - `GGUF.*`：reader 暴露的伪字段（version / tensor_count / kv_count），
    #      不是真的 KV，拷贝会写出重复项
    n_kv = 0
    for name, field in reader.fields.items():
        if name == "general.architecture" or name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), vtype, sub_type=sub_type)
        n_kv += 1

    # 2) 张量：跳过专家
    n_keep = 0
    n_drop = 0
    dropped_bytes = 0
    for t in reader.tensors:
        if t.name in drop:
            n_drop += 1
            dropped_bytes += t.n_bytes
            continue
        # raw_dtype 表示「按原始量化字节搬运」，不做反量化/再量化。
        # raw_shape 必须给**字节形状**（`t.data.shape`）—— gguf-py 会据此反推回
        # 量化形状写进 tensor info。给原始 ne 会报 "bytes per row is not a multiple"。
        # 与 gguf-py 自带的 gguf_editor_gui.py 用法一致。
        writer.add_tensor(t.name, t.data, raw_shape=t.data.shape,
                          raw_dtype=t.tensor_type, tensor_endianess=reader.endianess)
        n_keep += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"KV: 搬运 {n_kv} 项")
    print(f"张量: 保留 {n_keep} 个，剥掉 {n_drop} 个（{dropped_bytes / 2**30:.2f} GiB）")
    print(f"输出 {out} ({out.stat().st_size / 2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

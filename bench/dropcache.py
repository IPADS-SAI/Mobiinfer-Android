#!/usr/bin/env python3
"""dropcache.py — 用 posix_fadvise(DONTNEED) 丢弃模型/专家文件的 page cache（无需 root）。

用法：python3 dropcache.py [目录或文件 ...]
默认丢弃 ~/csm/moe-exp/models 下的所有 gguf 与 experts-htp-tiled/*.bin。
用于让每次 benchmark 从相同的冷缓存状态开始。
"""
import os
import sys
import glob

def drop(path, chunk=0):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        print(f"skip {path}: {e}", file=sys.stderr)
        return 0
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return 1
    finally:
        os.close(fd)

def main():
    targets = sys.argv[1:]
    if not targets:
        root = os.path.expanduser("~/csm/moe-exp/models/MiniCPM-V-4.7-35B-A3B-gguf-hexagon")
        targets = glob.glob(root + "/*.gguf") + glob.glob(root + "/experts-htp-tiled/*.bin")
    n = 0
    total = 0
    for t in targets:
        for f in (glob.glob(t) if any(c in t for c in "*?[") else [t]):
            n += drop(f)
            try:
                total += os.path.getsize(f)
            except OSError:
                pass
    print(f"dropped {n} files, {total/1e9:.2f} GB")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply (or dry-run) a partial Q3_K -> Q4_K layer upgrade to a Hexagon tiled expert pack.

The pack is the on-device `experts-htp-tiled` directory: 40 x layer_XXX.bin plus a
manifest.json that the runtime reads for per-tensor quant types and expert sizes.
Replacing a layer therefore needs *two* things, in this order:

  1. manifest.json patched for the changed layers (quant_type / bytes_per_expert /
     exact_byte_length / expert_stride / repacked_offset), written atomically;
  2. each layer_XXX.bin replaced by the pre-packed Q4_K (or mixed-family) file.

Order matters: a new manifest over old layer files makes the runtime fail loudly
("file is too small"), whereas a new layer file under the old manifest would make it
read the wrong strides silently.  So the manifest goes first and is always backed up.

Default is dry-run; nothing is written unless --apply is given.  Every mutation is
journaled to <pack>/.partial_q4k/journal.jsonl.  With the default --keep-bak the old
layer file is kept as layer_XXX.bin.bak so --rollback can restore the previous state;
--prune-bak deletes each .bak right after that layer verifies, which bounds the extra
disk to one layer but means --rollback can only restore the layers whose .bak survives.

stdlib only.  Usage:

  python3 apply_partial_q4k.py --plan partial_q4k_plan.json --base-dir DIR \
      --source-dir NEW_TILED_DIR                     # dry-run, prints the plan
  ... --apply [--prune-bak]                          # really replace
  ... --rollback                                     # undo journaled replacements
  ... --verify                                       # re-hash installed pack vs plan
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

JOURNAL_DIR = ".partial_q4k"
JOURNAL = "journal.jsonl"
CHUNK = 8 * 1024 * 1024


def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: str, limit: int | None = None) -> str:
    h = hashlib.sha256()
    left = limit
    with open(path, "rb") as fh:
        while True:
            n = CHUNK if left is None else min(CHUNK, left)
            if n <= 0:
                break
            b = fh.read(n)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def layer_name(layer: int) -> str:
    return f"layer_{layer:03d}.bin"


def journal_append(pack: str, record: dict):
    d = os.path.join(pack, JOURNAL_DIR)
    os.makedirs(d, exist_ok=True)
    rec = dict(record, ts=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(os.path.join(d, JOURNAL), "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def journal_rotate(pack: str):
    """Start a new apply session: archive the previous journal (audit trail)."""
    path = os.path.join(pack, JOURNAL_DIR, JOURNAL)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        os.replace(path, path + f".{stamp}.prev")


def journal_retire(pack: str, suffix: str):
    path = os.path.join(pack, JOURNAL_DIR, JOURNAL)
    if os.path.exists(path):
        os.replace(path, path + f".{time.strftime('%Y%m%dT%H%M%S')}.{suffix}")


def journal_read(pack: str) -> list[dict]:
    path = os.path.join(pack, JOURNAL_DIR, JOURNAL)
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def load_plan(path: str) -> dict:
    plan = json.load(open(path))
    if plan.get("format") != "partial-q4k-plan-v1":
        die(f"{path}: unsupported format {plan.get('format')!r}")
    for key in ("base_pack", "layers", "expected_delta_bytes", "manifest"):
        if key not in plan:
            die(f"{path}: missing key {key!r}")
    seen = set()
    for item in plan["layers"]:
        for key in ("layer", "families", "old_sha256", "new_sha256",
                    "old_bytes", "new_bytes", "delta_bytes", "manifest_entries"):
            if key not in item:
                die(f"layer entry missing {key!r}: {item}")
        if item["layer"] in seen:
            die(f"duplicate layer {item['layer']}")
        seen.add(item["layer"])
        if item["new_bytes"] - item["old_bytes"] != item["delta_bytes"]:
            die(f"layer {item['layer']}: delta_bytes != new_bytes - old_bytes")
    if sum(i["delta_bytes"] for i in plan["layers"]) != plan["expected_delta_bytes"]:
        die("expected_delta_bytes != sum(layer delta_bytes)")
    return plan


def patch_manifest(manifest: dict, plan: dict) -> dict:
    """Return a copy of manifest with entries for changed tensors replaced."""
    new = json.loads(json.dumps(manifest))  # deep copy, stdlib only
    by_name = {}
    for item in plan["layers"]:
        for entry in item["manifest_entries"]:
            by_name[entry["tensor_name"]] = entry
    replaced = set()
    entries = new.get("entries", [])
    for idx, entry in enumerate(entries):
        name = entry.get("tensor_name")
        if name in by_name:
            entries[idx] = by_name[name]
            replaced.add(name)
    missing = set(by_name) - replaced
    if missing:
        die(f"manifest has no entry for {sorted(missing)}")
    conv = new.get("hexagon_conversion")
    if isinstance(conv, dict) and isinstance(conv.get("files_sha256"), dict):
        files = conv["files_sha256"]
        for item in plan["layers"]:
            files[layer_name(item["layer"])] = item["new_sha256"]
        conv["files_sha256"] = dict(sorted(files.items()))
    return new


def resolve_source(item: dict, source_dir: str | None) -> str:
    if source_dir:
        return os.path.join(source_dir, layer_name(item["layer"]))
    path = item.get("source_path")
    if not path:
        die(f"layer {item['layer']}: no source_path and no --source-dir")
    return path


def preflight(plan: dict, args, expect_old: bool = True) -> list[dict]:
    base = os.path.abspath(os.path.expanduser(args.base_dir or plan["base_pack"]))
    if not os.path.isdir(base):
        die(f"base pack directory not found: {base}")
    manifest_path = os.path.join(base, plan["manifest"].get("path", "manifest.json"))
    if not os.path.isfile(manifest_path):
        die(f"base manifest not found: {manifest_path}")
    manifest = json.load(open(manifest_path))
    old_manifest_sha = sha256_file(manifest_path)
    if expect_old and plan["manifest"].get("old_sha256") and not args.skip_verify:
        if old_manifest_sha != plan["manifest"]["old_sha256"]:
            die(f"base manifest sha256 mismatch: {old_manifest_sha} != {plan['manifest']['old_sha256']}")

    rows = []
    for item in plan["layers"]:
        name = layer_name(item["layer"])
        dst = os.path.join(base, name)
        src = resolve_source(item, args.source_dir)
        if not os.path.isfile(dst):
            die(f"missing base layer file: {dst}")
        if not os.path.isfile(src):
            die(f"missing source layer file: {src}")
        size = os.path.getsize(dst)
        want = item["old_bytes"] if expect_old else item["new_bytes"]
        if size != want:
            die(f"{dst}: size {size} != expected {want}")
        nsize = os.path.getsize(src)
        if nsize != item["new_bytes"]:
            die(f"{src}: size {nsize} != new_bytes {item['new_bytes']}")
        row = dict(item, name=name, dst=dst, src=src)
        if not args.skip_verify:
            row["old_sha_ok"] = (sha256_file(dst) == item["old_sha256"]) if expect_old else None
            row["new_sha_ok"] = sha256_file(src) == item["new_sha256"]
        else:
            row["old_sha_ok"] = row["new_sha_ok"] = None
        rows.append(row)

    plan["_base"] = base
    plan["_manifest_path"] = manifest_path
    plan["_manifest"] = manifest
    plan["_old_manifest_sha"] = old_manifest_sha
    return rows


def free_bytes(path: str) -> int:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def print_rows(rows: list[dict]):
    print(f"{'layer':>5} {'old':>12} {'new':>12} {'delta':>12} "
          f"{'old_sha':>7} {'new_sha':>7}  families")
    for r in rows:
        fam = "+".join(f.replace("ffn_", "").replace("_exps", "") for f in r["families"])
        o = "-" if r["old_sha_ok"] is None else ("ok" if r["old_sha_ok"] else "MISMATCH")
        n = "-" if r["new_sha_ok"] is None else ("ok" if r["new_sha_ok"] else "MISMATCH")
        print(f"{r['layer']:>5} {human(r['old_bytes']):>12} {human(r['new_bytes']):>12} "
              f"{human(r['delta_bytes']):>12} {o:>7} {n:>7}  {fam}")


def cmd_apply(rows, plan, args):
    base = plan["_base"]
    keep_bak = not args.prune_bak
    manifest = plan["_manifest"]
    new_manifest = patch_manifest(manifest, plan)
    # Staged copies are consumed one layer at a time.  With .bak kept, the old
    # changed bytes stay around; with --prune-bak only the cumulative delta does.
    biggest = max(r["new_bytes"] for r in rows) if rows else 0
    need = biggest + sum(r["delta_bytes"] for r in rows)
    if keep_bak:
        need += sum(r["old_bytes"] for r in rows)
    free = free_bytes(base)
    print(f"space: need ~{human(need)} free, have {human(free)} at {base}")
    if free < need + (64 << 20):
        die("not enough free space; use --prune-bak or free space first "
            f"(need {human(need + (64 << 20))})", 3)

    work = os.path.join(base, JOURNAL_DIR)
    os.makedirs(work, exist_ok=True)
    journal_rotate(base)

    # 1) manifest first (fails safe: runtime errors on too-small files)
    mpath = plan["_manifest_path"]
    tmp_m = os.path.join(work, "manifest.json.new")
    with open(tmp_m, "w") as fh:
        json.dump(new_manifest, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    new_sha = sha256_file(tmp_m)
    if plan["manifest"].get("new_sha256") and new_sha != plan["manifest"]["new_sha256"]:
        die(f"patched manifest sha256 {new_sha} != plan {plan['manifest']['new_sha256']}")
    mbackup = mpath + ".bak"
    if not os.path.exists(mbackup):
        shutil.copy2(mpath, mbackup)
    os.replace(tmp_m, mpath)
    journal_append(base, {"op": "manifest", "path": "manifest.json",
                          "old_sha256": plan["_old_manifest_sha"], "new_sha256": new_sha})
    print(f"manifest patched ({len(rows)} layers), backup {os.path.basename(mbackup)}")

    # 2) layer files, one at a time
    done = []
    try:
        for r in rows:
            stage = os.path.join(work, r["name"] + ".new")
            shutil.copyfile(r["src"], stage)
            if sha256_file(stage) != r["new_sha256"]:
                die(f"staged copy sha256 mismatch for {r['name']}")
            bak = r["dst"] + ".bak"
            if os.path.exists(bak):
                die(f"refusing to overwrite existing {bak}")
            os.replace(r["dst"], bak)
            os.replace(stage, r["dst"])
            got = sha256_file(r["dst"])
            if got != r["new_sha256"]:
                os.replace(bak, r["dst"])
                die(f"{r['name']}: installed sha256 {got} != plan; layer restored")
            journal_append(base, {"op": "layer", "layer": r["layer"], "file": r["name"],
                                  "old_sha256": r["old_sha256"], "new_sha256": r["new_sha256"],
                                  "bak": os.path.basename(bak) if keep_bak else None})
            done.append(r)
            print(f"applied {r['name']}  {human(r['delta_bytes'])}")
            if not keep_bak:
                os.unlink(bak)
    except BaseException:
        print("failure: rolling back applied layers", file=sys.stderr)
        for r in reversed(done):
            bak = r["dst"] + ".bak"
            if os.path.exists(bak):
                os.replace(bak, r["dst"])
        if os.path.exists(mbackup):
            os.replace(mbackup, mpath)
        raise

    # 3) verify whole pack against plan
    bad = [r["name"] for r in rows if sha256_file(r["dst"]) != r["new_sha256"]]
    if bad:
        die(f"post-apply verification failed for {bad}")
    total = sum(r["delta_bytes"] for r in rows)
    print(f"OK: {len(rows)} layers applied, delta {human(total)} ({total} B)")
    if keep_bak:
        print("old layers kept as *.bak; run --rollback to undo")
    else:
        print("--prune-bak: old layer files removed; rollback only for surviving .bak")
    return 0


def cmd_rollback(plan, args):
    base = os.path.abspath(os.path.expanduser(args.base_dir or plan["base_pack"]))
    records = journal_read(base)
    if not records:
        die(f"no journal under {base}/{JOURNAL_DIR}")
    layers = [r for r in records if r.get("op") == "layer"]
    man = [r for r in records if r.get("op") == "manifest"]
    restored, missing = [], []
    for r in reversed(layers):
        dst = os.path.join(base, r["file"])
        bak = dst + ".bak"
        if os.path.exists(bak):
            os.replace(bak, dst)
            restored.append(r["file"])
        else:
            missing.append(r["file"])
    mpath = os.path.join(base, "manifest.json")
    if os.path.exists(mpath + ".bak"):
        os.replace(mpath + ".bak", mpath)
        man_restored = True
    else:
        man_restored = False
    journal_append(base, {"op": "rollback", "restored": restored, "missing_bak": missing,
                          "manifest_restored": man_restored})
    journal_retire(base, "rolledback")
    print(f"rollback: {len(restored)} layers restored, {len(missing)} without .bak, "
          f"manifest_bak={man_restored}")
    if missing:
        print("re-transfer these layers from the original pack: " + " ".join(missing))
        return 1
    return 0


def cmd_verify(plan, args):
    rows = preflight(plan, args, expect_old=False)
    bad = [r["name"] for r in rows if sha256_file(r["dst"]) != r["new_sha256"]]
    if bad:
        print(f"installed pack does NOT match plan for {len(bad)} layers: {bad[:5]} ...")
        return 1
    want = plan["manifest"].get("new_sha256")
    if want is None:
        print(f"OK: {len(rows)} layers match plan new_sha256; manifest sha not pinned in plan")
        return 0
    man_ok = sha256_file(plan["_manifest_path"]) == want
    print(f"OK: {len(rows)} layers match plan new_sha256; manifest match {man_ok}")
    return 0 if man_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--base-dir", help="target experts-htp-tiled dir (default: plan.base_pack)")
    ap.add_argument("--source-dir", help="dir holding new layer_XXX.bin (default: plan source_path)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--prune-bak", action="store_true",
                    help="delete each .bak after that layer verifies (bounds disk, weaker rollback)")
    ap.add_argument("--rollback", action="store_true", help="restore from journal/.bak")
    ap.add_argument("--verify", action="store_true", help="only verify installed pack vs plan")
    ap.add_argument("--skip-verify", action="store_true", help="skip sha256 preflight (fast, unsafe)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    plan = load_plan(args.plan)
    if args.rollback:
        return cmd_rollback(plan, args)
    if args.verify:
        return cmd_verify(plan, args)
    rows = preflight(plan, args)

    print(f"plan {args.plan}: base={plan['_base']}")
    print(f"budget {plan.get('budget_gib')} GiB, expected delta "
          f"{human(plan['expected_delta_bytes'])}, {len(rows)} layers")
    print_rows(rows)
    delta = sum(r["delta_bytes"] for r in rows)
    print(f"total delta: {human(delta)} ({delta} B); budget check: "
          f"{'OK' if delta <= plan.get('budget_bytes', 1 << 62) else 'OVER BUDGET'}")
    ok = all(r["old_sha_ok"] in (True, None) and r["new_sha_ok"] in (True, None) for r in rows)
    print(f"hashes: {'OK' if ok else 'MISMATCH'}")
    if not ok:
        return 1
    if args.json:
        print(json.dumps({"layers": len(rows), "delta_bytes": delta,
                          "dry_run": not args.apply, "hash_ok": ok}))
    if not args.apply:
        print("DRY-RUN: nothing written (pass --apply to replace)")
        return 0
    return cmd_apply(rows, plan, args)


if __name__ == "__main__":
    raise SystemExit(main())

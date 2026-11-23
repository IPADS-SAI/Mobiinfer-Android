#!/usr/bin/env python3
"""Build partial_q4k_plan.json from a packed final-arm tiled dir + the phone's current pack.

Inputs:
  --arm-pack DIR        tiled pack produced by tools/moe/pack_hexagon.py for the final arm
  --old-manifest FILE   copy of the phone's experts-htp-tiled/manifest.json
  --old-sums FILE       copy of the phone's SHA256SUMS (lines "<sha>  experts-htp-tiled/layer_XXX.bin")
  --select JSON         {"layers":[{"layer":33,"families":["ffn_gate_exps",...]}, ...]}
  --budget-gib 2.5
Outputs the plan JSON on stdout (and --out).
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys

TILED = {"Q3_K": 524288, "Q4_K": 655360, "Q6_K": 917504}
RAW = {"Q3_K": 450560, "Q4_K": 589824, "Q6_K": 860160}
FAM = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-pack", required=True)
    ap.add_argument("--old-manifest", required=True)
    ap.add_argument("--old-sums", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--budget-gib", type=float, required=True)
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    pack = os.path.abspath(args.arm_pack)
    new_man = json.load(open(os.path.join(pack, "manifest.json")))
    old_man = json.load(open(args.old_manifest))
    ent_by_name = {e["tensor_name"]: e for e in new_man["entries"]}
    old_by_name = {e["tensor_name"]: e for e in old_man["entries"]}
    old_sums = {}
    for line in open(args.old_sums):
        parts = line.split()
        if len(parts) == 2 and parts[1].endswith(".bin"):
            old_sums[os.path.basename(parts[1])] = parts[0]
    files_sha = new_man.get("hexagon_conversion", {}).get("files_sha256", {})
    sel = json.load(open(args.select))

    layers, total_delta, ram_delta = [], 0, 0
    for item in sorted(sel["layers"], key=lambda x: x["layer"]):
        layer = int(item["layer"])
        fams = item["families"]
        name = f"layer_{layer:03d}.bin"
        path = os.path.join(pack, name)
        nb = os.path.getsize(path)
        if files_sha.get(name) and sha256_file(path) != files_sha[name]:
            sys.exit(f"{name}: sha256 differs from pack manifest")
        # all three family entries must come from the new pack so offsets/stride stay consistent
        entries = []
        for f in FAM:
            tn = f"blk.{layer}.{f}.weight"
            e = ent_by_name[tn]
            if f in fams:
                if e["quant_type"] != "Q4_K":
                    sys.exit(f"{tn}: expected Q4_K in final arm, got {e['quant_type']}")
            else:
                if e["quant_type"] != old_by_name[tn]["quant_type"]:
                    sys.exit(f"{tn}: unchanged family type {e['quant_type']} != phone "
                             f"{old_by_name[tn]['quant_type']}")
            entries.append(e)
        want_bytes = 256 * sum(TILED[e["quant_type"]] for e in entries)
        if want_bytes != nb:
            sys.exit(f"{name}: size {nb} != 256*sum(tiled) {want_bytes}")
        ob = 256 * sum(TILED[old_by_name[f'blk.{layer}.{f}.weight']["quant_type"]] for f in FAM)
        old_rows = [old_by_name[f"blk.{layer}.{f}.weight"] for f in FAM]
        if old_by_name[f"blk.{layer}.{FAM[0]}.weight"]["expert_stride"] != sum(o["bytes_per_expert"] for o in old_rows):
            sys.exit(f"{name}: phone pack layout is not 3-family contiguous")
        ram_delta += sum(e["bytes_per_expert"] - o["bytes_per_expert"] for e, o in zip(entries, old_rows))
        layers.append(dict(
            layer=layer, families=fams,
            old_sha256=old_sums[name], new_sha256=files_sha[name],
            old_bytes=ob, new_bytes=nb, delta_bytes=nb - ob,
            source_path=path,
            manifest_entries=entries,
        ))
        total_delta += nb - ob

    old_man_sha = sha256_file(args.old_manifest)
    # patched manifest must be byte-identical to what apply_partial_q4k.py will write
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from apply_partial_q4k import patch_manifest
    plan_stub = dict(layers=layers)
    new_man_patched = patch_manifest(old_man, plan_stub)
    blob = json.dumps(new_man_patched, indent=2) + "\n"
    new_man_sha = hashlib.sha256(blob.encode()).hexdigest()

    base_files = sorted(n for n in os.listdir(pack) if n.endswith(".bin"))
    base_bytes = sum(256 * sum(TILED[old_by_name[f"blk.{l}.{f}.weight"]["quant_type"]] for f in FAM)
                     for l in range(40))
    plan = dict(
        format="partial-q4k-plan-v1",
        base_pack="/data/data/com.termux/files/home/csm/moe-exp/models/"
                  "MiniCPM-V-4.7-35B-A3B-gguf-hexagon/experts-htp-tiled",
        base_sha256=old_man_sha,
        base_layer_count=40,
        base_layer_bytes=402653184,
        base_experts_bytes=base_bytes,
        base_experts_gib=round(base_bytes / 2**30, 4),
        budget_gib=args.budget_gib,
        budget_bytes=int(args.budget_gib * 2**30),
        expected_delta_bytes=total_delta,
        expected_delta_gib=round(total_delta / 2**30, 4),
        resulting_experts_bytes=base_bytes + total_delta,
        resulting_experts_gib=round((base_bytes + total_delta) / 2**30, 4),
        ram_delta_bytes_per_slot=ram_delta,
        layers=layers,
        manifest=dict(path="manifest.json", old_sha256=old_man_sha, new_sha256=new_man_sha,
                      sha256sums_old="experts-htp-tiled/manifest.json"),
        rollback=dict(
            strategy="journal+bak",
            journal=".partial_q4k/journal.jsonl",
            keep_bak_default=True,
            prune_bak_flag=True,
            command="python3 apply_partial_q4k.py --plan partial_q4k_plan.json --rollback",
            note="--prune-bak bounds peak disk to one layer but only surviving .bak can be "
                 "rolled back; full rollback needs >= old_bytes(changed) free.",
            required_free_bytes_keep_bak=total_delta + max(i["new_bytes"] for i in layers)
                                          + sum(i["old_bytes"] for i in layers),
            required_free_bytes_prune_bak=total_delta + max(i["new_bytes"] for i in layers),
        ),
        notes=dict(
            source_arm=new_man.get("source", {}),
            arm_pack=pack,
            files_sha256=files_sha,
            quant_bytes=dict(raw=RAW, tiled=TILED),
        ),
    )
    out = json.dumps(plan, indent=2) + "\n"
    if args.out == "-":
        sys.stdout.write(out)
    else:
        open(args.out, "w").write(out)
        print(f"wrote {args.out}: {len(layers)} layers, delta "
              f"{total_delta} B ({total_delta/2**30:.4f} GiB), budget {args.budget_gib} GiB",
              file=sys.stderr)


if __name__ == "__main__":
    main()

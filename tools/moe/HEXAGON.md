# Prepare externally stored experts for Hexagon

The tools operate on existing GGUF weights; they do not train the model.
Run from the repository root with Python 3.10+ and the dependencies in
`gguf-py/requirements.txt` installed. Keep the original model until the
converted deployment has been verified.

## Extract experts and strip their duplicate GGUF data

These two tools already exist in this repository:

```sh
python tools/moe/experts.py extract --model /models/input.gguf --out-dir /models/experts-raw
python tools/moe/experts.py verify --model /models/input.gguf --experts /models/experts-raw
python tools/moe/strip_experts.py --model /models/input.gguf --experts /models/experts-raw --out /models/model-les.gguf
```

The manifest retains the expert shape, quantization and original eight-expert
routing metadata. The stripped GGUF keeps the resident tensors; it must be
used together with its matching expert manifest/files. `strip_experts.py`
expects a single GGUF input; merge split GGUFs before using that tool.

## Pack the experts for HTP

Use the Hexagon libraries built from the same revision as the deployed runner.
The Python process must have the same architecture as these libraries. For
Android libraries, run the command on the Android device with Python/NumPy
and this repository's Python tools available. Ensure transitive native library
and DSP skeleton dependencies can be found via the normal library paths.

```sh
python tools/moe/pack_hexagon.py \
    --experts /models/experts-raw \
    --out-dir /models/experts-htp-tiled \
    --lib-dir /path/to/matching/lib
```

This tool accepts expert types supported by the matching backend, including
mixed types across layers. It calls the backend's `ggml_backend_pack_moe_expert`
API (and `ggml_backend_pack_moe_expert_compact` with `--compact-q3k`), so there
is no second Python implementation of the tiled format. It retains the raw
manifest's shape/type/stride fields, marks the backend layout as
`hexagon-tiled-v1` (or `hexagon-tiled-q3k-compact-v1`, see below), and records
output file SHA256 values, the backend hash and per-expert CRC32 values in
`hexagon_conversion.expert_checksums`. The slot reader checks those CRCs before
making a loaded expert available to the graph; a mismatch aborts the graph
instead of feeding the expert to the kernels. Set `LLAMA_MOE_VERIFY_CRC=0` to
skip the per-read check (diagnostics only, and the log then reports
`packed-crc: checked=0`).

Pass `--compact-q3k` with a runner that supports the
`hexagon-tiled-q3k-compact-v1` layout. The compact format keeps Q3_K as raw
110-byte blocks (one contiguous 32-row slab, 128-byte aligned at the slab end)
instead of the 512-byte/32-element tiled tile, which is smaller but understood
only by a matching DSP skeleton; a mismatch fails model loading with
`backend cannot allocate compact Q3_K expert tensor` rather than silently
reading the wrong bytes. Other quantizations keep their standard tile format.
Use the resulting directory for both `--moe-experts` and
`LLAMA_MOE_HTP_PACKED_DIR`. Older standard-layout manifests remain readable
(they simply have no checksum index, so no CRC is checked), and a compact
manifest is rejected by older runners with `unsupported backend expert layout`.

`GGML_HEXAGON_MOE_HMX_PIPELINE=1` enables same-layer DMA, HVX and HMX
overlap for supported routed matrix multiplications. It is the only validated
path for compact Q3_K; `GGML_HEXAGON_MOE_HMX_PIPELINE=0 GGML_HEXAGON_MOE_HMX_SERIAL=1`
is an opt-in escape hatch that runs compact experts through the legacy serial
HMX skeleton instead (not yet validated bit-exactly on device - keep it for A/B
only). Runtime code computes tiled offsets independently of the raw offsets.

The destination must be empty. Files are written atomically, and the usable
manifest is published only after all layers complete. Interrupted conversions
leave completed files without a final manifest: use a new empty output directory
for a retry. Existing files are not accepted based on size alone.

## Shared expert-bank decode (opt-in)

`LLAMA_MOE_SHARED_DECODE=1` lets all routed layers draw slots from one shared
bank with a single ClassLRU map instead of one decode window per layer. It is
demand-only: it targets single-row decode with the whole bank available, and it
trades the prefetch/eviction machinery away for a larger cache.

Hard requirements (each one is validated at model load or context creation, with
an error naming the offending knob):

- `LLAMA_MOE_DECODE_WINDOW=0` - the shared map replaces the per-layer window.
- `LLAMA_MOE_LOOKAHEAD_PREFETCH=0` and `LLAMA_MOE_MATMUL_SPLIT=0` - there is no
  window to prefetch into and no phase split to defer reads into.
  (`moe_decode_phase` is derived from `decode_window > 0`, so the matmul split
  is unreachable in this mode by construction.)
- one `llama_context` per model - the bank lease is exclusive. Servers that
  create several contexts per model will fail on the second one.
- `--moe-slot-bank` must exceed the whole decoded working set, because every
  slot is shared by all routed layers: the working set is roughly
  `routed_layers * expert_used` (40 x 8 = 320 for MiniCPM-V 4.7), so a bank of
  256 slots thrashes (measured: 2.3x the misses of a 3200-slot bank, 4.9 t/s vs
  7.9 t/s on the same prompt). Size the bank to the working set, not to one
  layer.
- a CRC-indexed packed manifest (see above) - the shared map assumes the packed
  slot layout, so a raw/layered expert directory is rejected at load time.
- the slot-manager knobs are rejected, not silently ignored: shared decode has
  its own ClassLRU, and `init_slot_managers()` never runs without a decode
  window, so `LLAMA_MOE_EXPERT_READY`, `LLAMA_MOE_EVICT=lfu*`,
  `LLAMA_MOE_EVICT_LAYERS`, `LLAMA_MOE_EXPERT_IO_BUDGET`,
  `LLAMA_MOE_PF_PROBATION` and `LLAMA_MOE_PREFETCH_H1_FRONT` would otherwise
  have no effect at all.

Speculative decoding (`--spec-type ...`) sends verification batches of
`n_max + 1` rows. Those are not single-row decodes, so the shared map is
invalidated on every such batch; the runtime logs a one-time warning when this
happens. Do not combine the two unless you have measured the resulting miss
rate.

Requirements are checked in two stages, and a configuration that breaks several
of them reports only the first stage's error:

1. **model load** (`load_hparams`): `LLAMA_MOE_DECODE_WINDOW` must be 0, the
   manifest must carry `expert_checksums` *and* `LLAMA_MOE_HTP_PACKED_DIR` must
   be set, then the bank size limit;
2. **context creation** (`init_shared_decode`): lookahead / matmul-split /
   window must already be off, the bank must be one aliased packed bank, and
   finally the slot-manager knobs above are rejected.

So `LLAMA_MOE_SHARED_DECODE=1` together with `LLAMA_MOE_EXPERT_READY=1` reports
the `EXPERT_READY` rejection (prefetch is off by default), and only if you also
set `LLAMA_MOE_LOOKAHEAD_PREFETCH=1` do you see the prefetch gate first.

The switch itself follows the usual boolean spelling: `0` / `false` / `off`
disable it, `1` / `true` / `on` enable it, unset means off, and any other
non-empty value is treated as enabled with a warning.

## Run

Transfer `model-les.gguf` and the complete tiled directory to the device.
The manifest and binaries must stay together. Set both the loader manifest
path and the tiled-path override to the tiled directory:

```sh
export LLAMA_MOE_HTP_PACKED_DIR=/models/experts-htp-tiled
./bin/llama-cli -m /models/model-les.gguf \
    --moe-experts /models/experts-htp-tiled --moe-slot-bank 32 \
    --device HTP0,HTP1 --split-mode layer --tensor-split 1,1 -ngl 99 \
    -fit off -c 2048 -b 4 -ub 4 -p 'Hello'
```

The tiled files cannot be used as raw experts or verified byte-for-byte with
`experts.py verify`; that command verifies the extraction before conversion.
Compare generated tiled checksums with independently verified deployment
assets when available. This tool does not convert the DSpark or vision GGUFs;
those are separate assets. Generated model/expert binaries do not belong in Git.

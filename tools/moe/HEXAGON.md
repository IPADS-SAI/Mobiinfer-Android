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

This tool supports Q3_K, Q4_K and Q6_K expert tensors, including mixed types
across layers. It calls the backend's `ggml_backend_pack_moe_expert` API, so
there is no second Python implementation of the tiled format. It retains the
raw manifest's shape/type/stride fields, marks the backend layout as
`hexagon-tiled-v1`, and records output file SHA256 values and the backend hash.
Runtime code computes tiled offsets independently of the raw offsets.

The destination must be empty. Files are written atomically, and the usable
manifest is published only after all layers complete. Interrupted conversions
leave completed files without a final manifest: use a new empty output directory
for a retry. Existing files are not accepted based on size alone.

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

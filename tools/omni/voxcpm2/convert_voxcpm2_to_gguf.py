#!/usr/bin/env python3
"""
Convert VoxCPM2 PyTorch weights to GGUF format for llama.cpp-omni.

Usage:
    python convert_voxcpm2_to_gguf.py \\
        --model /path/to/MiniCPM-o-4_5-gguf/model.safetensors \\
        --vae /path/to/MiniCPM-o-4_5-gguf/audiovae.pth \\
        --config /path/to/MiniCPM-o-4_5-gguf/config.json \\
        --output /path/to/output_dir

Output:
    output_dir/
    ├── VoxCPM2-BaseLM-F16.gguf       # arch=minicpm (28-layer causal decoder)
    └── VoxCPM2-Acoustic-F16.gguf     # arch=voxcpm2-acoustic (ResidualLM + LocEnc + LocDiT + FSQ + AudioVAE + Projections)
"""

import argparse
import json
import os
import struct
import sys
import numpy as np

# Add llama.cpp gguf-py to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "..")
GGUF_PY_DIR = os.path.join(REPO_ROOT, "gguf-py")
sys.path.insert(0, GGUF_PY_DIR)

import gguf
from gguf import GGUFWriter, GGMLQuantizationType, Keys, RopeScalingType


# ==============================================================================
# PyTorch → GGUF tensor name mapping for BaseLM (LLM_ARCH_MINICPM)
# ==============================================================================
BASE_LM_GGUF_TENSOR_MAP = {
    # Embedding & norm
    "base_lm.embed_tokens.weight": "token_embd.weight",
    "base_lm.norm.weight": "output_norm.weight",
    # lm_head tied with embed_tokens → output.weight (duplicated)
    # Per-layer
    "base_lm.layers.{i}.input_layernorm.weight": "blk.{i}.attn_norm.weight",
    "base_lm.layers.{i}.self_attn.q_proj.weight": "blk.{i}.attn_q.weight",
    "base_lm.layers.{i}.self_attn.k_proj.weight": "blk.{i}.attn_k.weight",
    "base_lm.layers.{i}.self_attn.v_proj.weight": "blk.{i}.attn_v.weight",
    "base_lm.layers.{i}.self_attn.o_proj.weight": "blk.{i}.attn_output.weight",
    "base_lm.layers.{i}.post_attention_layernorm.weight": "blk.{i}.ffn_norm.weight",
    "base_lm.layers.{i}.mlp.gate_proj.weight": "blk.{i}.ffn_gate.weight",
    "base_lm.layers.{i}.mlp.up_proj.weight": "blk.{i}.ffn_up.weight",
    "base_lm.layers.{i}.mlp.down_proj.weight": "blk.{i}.ffn_down.weight",
}

# ==============================================================================
# PyTorch → GGUF tensor name mapping for Acoustic components
# ==============================================================================
ACOUSTIC_GGUF_PREFIX_MAP = {
    # ResidualLM (8 layers, no_rope)
    "residual_lm.norm.weight": "residual_lm.output_norm.weight",
    "residual_lm.layers.{i}.input_layernorm.weight": "residual_lm.blk.{i}.attn_norm.weight",
    "residual_lm.layers.{i}.self_attn.q_proj.weight": "residual_lm.blk.{i}.attn_q.weight",
    "residual_lm.layers.{i}.self_attn.k_proj.weight": "residual_lm.blk.{i}.attn_k.weight",
    "residual_lm.layers.{i}.self_attn.v_proj.weight": "residual_lm.blk.{i}.attn_v.weight",
    "residual_lm.layers.{i}.self_attn.o_proj.weight": "residual_lm.blk.{i}.attn_output.weight",
    "residual_lm.layers.{i}.post_attention_layernorm.weight": "residual_lm.blk.{i}.ffn_norm.weight",
    "residual_lm.layers.{i}.mlp.gate_proj.weight": "residual_lm.blk.{i}.ffn_gate.weight",
    "residual_lm.layers.{i}.mlp.up_proj.weight": "residual_lm.blk.{i}.ffn_up.weight",
    "residual_lm.layers.{i}.mlp.down_proj.weight": "residual_lm.blk.{i}.ffn_down.weight",
    # LocEnc (12 layers, non-causal)
    "feat_encoder.encoder.norm.weight": "locenc.norm.weight",
    "feat_encoder.encoder.layers.{i}.input_layernorm.weight": "locenc.blk.{i}.attn_norm.weight",
    "feat_encoder.encoder.layers.{i}.self_attn.q_proj.weight": "locenc.blk.{i}.attn_q.weight",
    "feat_encoder.encoder.layers.{i}.self_attn.k_proj.weight": "locenc.blk.{i}.attn_k.weight",
    "feat_encoder.encoder.layers.{i}.self_attn.v_proj.weight": "locenc.blk.{i}.attn_v.weight",
    "feat_encoder.encoder.layers.{i}.self_attn.o_proj.weight": "locenc.blk.{i}.attn_output.weight",
    "feat_encoder.encoder.layers.{i}.post_attention_layernorm.weight": "locenc.blk.{i}.ffn_norm.weight",
    "feat_encoder.encoder.layers.{i}.mlp.gate_proj.weight": "locenc.blk.{i}.ffn_gate.weight",
    "feat_encoder.encoder.layers.{i}.mlp.up_proj.weight": "locenc.blk.{i}.ffn_up.weight",
    "feat_encoder.encoder.layers.{i}.mlp.down_proj.weight": "locenc.blk.{i}.ffn_down.weight",
    "feat_encoder.in_proj.weight": "locenc.in_proj.weight",
    "feat_encoder.in_proj.bias": "locenc.in_proj.bias",
    "feat_encoder.special_token": "locenc.cls_token.weight",
    # LocDiT (12 layers, non-causal + CFG)
    "feat_decoder.estimator.decoder.norm.weight": "locdit.norm.weight",
    "feat_decoder.estimator.decoder.layers.{i}.input_layernorm.weight": "locdit.blk.{i}.attn_norm.weight",
    "feat_decoder.estimator.decoder.layers.{i}.self_attn.q_proj.weight": "locdit.blk.{i}.attn_q.weight",
    "feat_decoder.estimator.decoder.layers.{i}.self_attn.k_proj.weight": "locdit.blk.{i}.attn_k.weight",
    "feat_decoder.estimator.decoder.layers.{i}.self_attn.v_proj.weight": "locdit.blk.{i}.attn_v.weight",
    "feat_decoder.estimator.decoder.layers.{i}.self_attn.o_proj.weight": "locdit.blk.{i}.attn_output.weight",
    "feat_decoder.estimator.decoder.layers.{i}.post_attention_layernorm.weight": "locdit.blk.{i}.ffn_norm.weight",
    "feat_decoder.estimator.decoder.layers.{i}.mlp.gate_proj.weight": "locdit.blk.{i}.ffn_gate.weight",
    "feat_decoder.estimator.decoder.layers.{i}.mlp.up_proj.weight": "locdit.blk.{i}.ffn_up.weight",
    "feat_decoder.estimator.decoder.layers.{i}.mlp.down_proj.weight": "locdit.blk.{i}.ffn_down.weight",
    "feat_decoder.estimator.in_proj.weight": "locdit.in_proj.weight",
    "feat_decoder.estimator.in_proj.bias": "locdit.in_proj.bias",
    "feat_decoder.estimator.out_proj.weight": "locdit.out_proj.weight",
    "feat_decoder.estimator.out_proj.bias": "locdit.out_proj.bias",
    "feat_decoder.estimator.cond_proj.weight": "locdit.cond_proj.weight",
    "feat_decoder.estimator.cond_proj.bias": "locdit.cond_proj.bias",
    "feat_decoder.estimator.time_mlp.linear_1.weight": "locdit.time_mlp.linear_1.weight",
    "feat_decoder.estimator.time_mlp.linear_1.bias": "locdit.time_mlp.linear_1.bias",
    "feat_decoder.estimator.time_mlp.linear_2.weight": "locdit.time_mlp.linear_2.weight",
    "feat_decoder.estimator.time_mlp.linear_2.bias": "locdit.time_mlp.linear_2.bias",
    "feat_decoder.estimator.delta_time_mlp.linear_1.weight": "locdit.delta_time_mlp.linear_1.weight",
    "feat_decoder.estimator.delta_time_mlp.linear_1.bias": "locdit.delta_time_mlp.linear_1.bias",
    "feat_decoder.estimator.delta_time_mlp.linear_2.weight": "locdit.delta_time_mlp.linear_2.weight",
    "feat_decoder.estimator.delta_time_mlp.linear_2.bias": "locdit.delta_time_mlp.linear_2.bias",
    # FSQ
    "fsq_layer.in_proj.weight": "fsq.in_proj.weight",
    "fsq_layer.in_proj.bias": "fsq.in_proj.bias",
    "fsq_layer.out_proj.weight": "fsq.out_proj.weight",
    "fsq_layer.out_proj.bias": "fsq.out_proj.bias",
    # Projections
    "enc_to_lm_proj.weight": "projections.enc_to_lm_proj.weight",
    "enc_to_lm_proj.bias": "projections.enc_to_lm_proj.bias",
    "lm_to_dit_proj.weight": "projections.lm_to_dit_proj.weight",
    "lm_to_dit_proj.bias": "projections.lm_to_dit_proj.bias",
    "res_to_dit_proj.weight": "projections.res_to_dit_proj.weight",
    "res_to_dit_proj.bias": "projections.res_to_dit_proj.bias",
    "fusion_concat_proj.weight": "projections.res_fusion_proj.weight",
    "fusion_concat_proj.bias": "projections.res_fusion_proj.bias",
    # StopPredictor
    "stop_proj.weight": "stop_predictor.linear1.weight",
    "stop_proj.bias": "stop_predictor.linear1.bias",
    "stop_head.weight": "stop_predictor.linear2.weight",
}

# ==============================================================================
# AudioVAE PyTorch weight_norm → GGUF tensor name mapping
# ==============================================================================
# AudioVAE uses weight_norm: weight = w_g * w_v / ||w_v||_2
# We need to merge w_g and w_v into a single weight tensor before writing.
# Keys ending in .weight_v will be merged with corresponding .weight_g.

def merge_weight_norm(weight_g: np.ndarray, weight_v: np.ndarray) -> np.ndarray:
    """Merge PyTorch weight_norm: weight = w_g * w_v / ||w_v||_2 (along dim=0 for Conv1d)"""
    # weight_g shape: [OC, 1, 1], weight_v shape: [OC, IC, K]
    # weight_norm dim=0 normalizes over all dims except 0
    v_norm = np.linalg.norm(weight_v.reshape(weight_v.shape[0], -1), axis=1, keepdims=True)
    v_norm = v_norm.reshape(weight_g.shape[0], 1, 1)  # [OC, 1, 1]
    merged = weight_g * weight_v / v_norm
    return merged


# AudioVAE mapping: PyTorch key (without .weight_g/.weight_v suffix) → GGUF key
def map_audiovae_key(pt_key: str) -> str:
    """Map AudioVAE pytorch key to GGUF key with audio_vae. prefix"""
    # Remove .module. prefix if present
    clean = pt_key.replace(".module.", ".")
    return f"audio_vae.{clean}"


# ==============================================================================
# Tensor type helpers
# ==============================================================================
def to_f32_if_norm(name: str, tensor: np.ndarray) -> np.ndarray:
    """Convert norm/rms weights to F32 to avoid ggml CPU backend f32×f16 type mismatch.
    The ggml binary_op only supports f16×f32→f32 but not f32×f16→f32.
    build_norm does ggml_mul(f32_rms_out, weight), so weight must be F32.
    """
    if 'norm' in name.lower() or 'rms' in name.lower():
        return tensor.astype(np.float32)
    return tensor


# ==============================================================================
# BF16 conversion helper
# ==============================================================================
def bf16_to_f16(data: np.ndarray) -> np.ndarray:
    """Convert BF16 numpy array to F16.

    BF16 is stored as uint16 in safetensors with dtype=BF16.
    We interpret the bytes as bfloat16 → float32 → float16.
    """
    if data.dtype == np.uint16:
        # Reinterpret uint16 as bf16 (top 16 bits of float32)
        as_uint32 = data.astype(np.uint32) << 16
        as_f32 = as_uint32.view(np.float32)
        return as_f32.astype(np.float16)
    elif data.dtype == np.float16:
        return data
    elif data.dtype == np.float32:
        return data.astype(np.float16)
    else:
        raise ValueError(f"Unsupported dtype for BF16 conversion: {data.dtype}")


# ==============================================================================
# Safetensors loader (works without torch)
# ==============================================================================
class SafeTensorFile:
    """Lightweight safetensors reader that doesn't require torch."""

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, 'rb')
        header_size = struct.unpack('<Q', self._file.read(8))[0]
        self.header = json.loads(self._file.read(header_size).decode('utf-8'))
        self._data_start = 8 + header_size

    def keys(self):
        return sorted(self.header.keys())

    def get_shape(self, key: str) -> list:
        return self.header[key]['shape']

    def get_dtype(self, key: str) -> str:
        return self.header[key]['dtype']

    def get_tensor(self, key: str) -> np.ndarray:
        if key not in self.header:
            raise KeyError(key)
        info = self.header[key]
        # Calculate data offset
        offset = self._data_start
        for k in sorted(self.header.keys()):
            if k == key:
                break
            tensor_start, tensor_end = self.header[k]['data_offsets']
            offset += tensor_end - tensor_start
        else:
            tensor_start, tensor_end = info['data_offsets']
            offset += tensor_start

        start, end = info['data_offsets']
        self._file.seek(offset)
        raw = self._file.read(end - start)

        shape = info['shape']
        dtype = info['dtype']

        if dtype == 'F32':
            return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
        elif dtype == 'F16':
            return np.frombuffer(raw, dtype=np.float16).reshape(shape).copy()
        elif dtype == 'BF16':
            data = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
            return bf16_to_f16(data)
        elif dtype == 'I32':
            return np.frombuffer(raw, dtype=np.int32).reshape(shape).copy()
        elif dtype == 'I64':
            return np.frombuffer(raw, dtype=np.int64).reshape(shape).copy()
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def close(self):
        self._file.close()


# ==============================================================================
# BaseLM GGUF writer
# ==============================================================================
def export_baselm(sf: SafeTensorFile, config: dict, output_path: str):
    """Export BaseLM (28-layer causal decoder) as standard LLM_ARCH_MINICPM GGUF."""
    lm_cfg = config['lm_config']
    n_layer = lm_cfg['num_hidden_layers']
    n_embd = lm_cfg['hidden_size']
    n_ff = lm_cfg['intermediate_size']
    n_head = lm_cfg['num_attention_heads']
    n_kv_head = lm_cfg['num_key_value_heads']
    n_embd_head_k = lm_cfg.get('kv_channels', n_embd // n_head)
    n_rot = n_embd_head_k
    rms_eps = lm_cfg.get('rms_norm_eps', 1e-5)
    rope_theta = lm_cfg.get('rope_theta', 10000.0)
    context_length = lm_cfg.get('max_position_embeddings', 32768)
    vocab_size = lm_cfg.get('vocab_size', 73448)
    scale_emb = lm_cfg.get('scale_emb', 12)
    dim_model_base = lm_cfg.get('dim_model_base', 256)
    scale_depth = lm_cfg.get('scale_depth', 1.4)
    use_mup = lm_cfg.get('use_mup', False)
    rope_scaling = lm_cfg.get('rope_scaling', None)

    # Compute derived params
    f_embedding_scale = float(scale_emb)
    f_residual_scale = scale_depth / np.sqrt(float(n_layer)) if not use_mup else 1.0
    f_logit_scale = dim_model_base / n_embd

    print(f"  BaseLM: {n_layer} layers, n_embd={n_embd}, n_head={n_head}, n_kv_head={n_kv_head}")
    print(f"          n_ff={n_ff}, vocab={vocab_size}, ctx={context_length}")
    print(f"          scale_emb={f_embedding_scale}, residual_scale={f_residual_scale:.4f}, logit_scale={f_logit_scale:.4f}")

    arch = "minicpm"
    gguf_writer = GGUFWriter(output_path, arch)

    # Metadata — Keys use {arch} placeholder, must format before passing
    gguf_writer.add_uint32(Keys.LLM.CONTEXT_LENGTH.format(arch=arch), context_length)
    gguf_writer.add_uint32(Keys.LLM.EMBEDDING_LENGTH.format(arch=arch), n_embd)
    gguf_writer.add_uint32(Keys.LLM.BLOCK_COUNT.format(arch=arch), n_layer)
    gguf_writer.add_uint32(Keys.LLM.FEED_FORWARD_LENGTH.format(arch=arch), n_ff)
    gguf_writer.add_uint32(Keys.LLM.VOCAB_SIZE.format(arch=arch), vocab_size)

    gguf_writer.add_uint32(Keys.Attention.HEAD_COUNT.format(arch=arch), n_head)
    gguf_writer.add_uint32(Keys.Attention.HEAD_COUNT_KV.format(arch=arch), n_kv_head)
    gguf_writer.add_float32(Keys.Attention.LAYERNORM_RMS_EPS.format(arch=arch), rms_eps)
    gguf_writer.add_uint32(Keys.Attention.KEY_LENGTH.format(arch=arch), n_embd_head_k)
    gguf_writer.add_uint32(Keys.Attention.VALUE_LENGTH.format(arch=arch), n_embd_head_k)

    gguf_writer.add_uint32(Keys.Rope.DIMENSION_COUNT.format(arch=arch), n_rot)
    gguf_writer.add_float32(Keys.Rope.FREQ_BASE.format(arch=arch), rope_theta)
    gguf_writer.add_bool(Keys.Rope.SCALING_FINETUNED.format(arch=arch), True)

    # MiniCPM-specific scales
    gguf_writer.add_float32(Keys.LLM.EMBEDDING_SCALE.format(arch=arch), f_embedding_scale)
    gguf_writer.add_float32(Keys.LLM.RESIDUAL_SCALE.format(arch=arch), f_residual_scale)
    gguf_writer.add_float32(Keys.LLM.LOGIT_SCALE.format(arch=arch), f_logit_scale)

    # LongRoPE scaling factors
    if rope_scaling and rope_scaling.get('type') == 'longrope':
        gguf_writer.add_string(Keys.Rope.SCALING_TYPE.format(arch=arch), RopeScalingType.LONGROPE.value)
        gguf_writer.add_uint32(Keys.Rope.SCALING_ORIG_CTX_LEN.format(arch=arch),
                               rope_scaling.get('original_max_position_embeddings', context_length))
        long_factors = rope_scaling.get('long_factor', None)
        short_factors = rope_scaling.get('short_factor', None)
    else:
        gguf_writer.add_string(Keys.Rope.SCALING_TYPE, RopeScalingType.NONE.value)
        long_factors = None
        short_factors = None

    # LongRoPE per-layer factors (64 head_dims/2 values per factor)
    n_rope_factors = n_rot // 2  # 64 for n_rot=128

    # Write tokenizer info (VoxCPM2 uses sentencepiece with special tokens)
    gguf_writer.add_string(Keys.Tokenizer.MODEL, "no_vocab")
    gguf_writer.add_uint32(Keys.Tokenizer.BOS_ID, lm_cfg.get('bos_token_id', 1))
    gguf_writer.add_uint32(Keys.Tokenizer.EOS_ID, lm_cfg.get('eos_token_id', 2))

    # Write tensors
    print("  Writing BaseLM tensors...")

    # Embedding
    embed = sf.get_tensor("base_lm.embed_tokens.weight")  # [vocab, embd]
    gguf_writer.add_tensor("token_embd.weight", embed)
    # lm_head tied with embedding → duplicate as output.weight
    gguf_writer.add_tensor("output.weight", embed)
    # Final norm (F32 to avoid ggml f32×f16 type mismatch in build_norm)
    gguf_writer.add_tensor("output_norm.weight",
                           to_f32_if_norm("output_norm.weight", sf.get_tensor("base_lm.norm.weight")))

    # Store LongRoPE factors as GGUF metadata (not tensors) to avoid per-layer
    # naming mismatch with llama.cpp's MiniCPM loader. VoxCPM2Transformer reads
    # them from these metadata keys, falling back to hardcoded defaults if absent.

    # Per-layer tensors
    for i in range(n_layer):
        pt_prefix = f"base_lm.layers.{i}"
        gguf_prefix = f"blk.{i}"

        gguf_writer.add_tensor(f"{gguf_prefix}.attn_norm.weight",
                               to_f32_if_norm("attn_norm", sf.get_tensor(f"{pt_prefix}.input_layernorm.weight")))
        gguf_writer.add_tensor(f"{gguf_prefix}.attn_q.weight",
                               sf.get_tensor(f"{pt_prefix}.self_attn.q_proj.weight"))
        gguf_writer.add_tensor(f"{gguf_prefix}.attn_k.weight",
                               sf.get_tensor(f"{pt_prefix}.self_attn.k_proj.weight"))
        gguf_writer.add_tensor(f"{gguf_prefix}.attn_v.weight",
                               sf.get_tensor(f"{pt_prefix}.self_attn.v_proj.weight"))
        gguf_writer.add_tensor(f"{gguf_prefix}.attn_output.weight",
                               sf.get_tensor(f"{pt_prefix}.self_attn.o_proj.weight"))
        gguf_writer.add_tensor(f"{gguf_prefix}.ffn_norm.weight",
                               to_f32_if_norm("ffn_norm", sf.get_tensor(f"{pt_prefix}.post_attention_layernorm.weight")))
        gguf_writer.add_tensor(f"{gguf_prefix}.ffn_gate.weight",
                               sf.get_tensor(f"{pt_prefix}.mlp.gate_proj.weight"))
        gguf_writer.add_tensor(f"{gguf_prefix}.ffn_up.weight",
                               sf.get_tensor(f"{pt_prefix}.mlp.up_proj.weight"))
        gguf_writer.add_tensor(f"{gguf_prefix}.ffn_down.weight",
                               sf.get_tensor(f"{pt_prefix}.mlp.down_proj.weight"))

        # Per-layer LongRoPE factors — DISABLED
        # llama.cpp loads global rope_factors for MINICPM; per-layer factors
        # cause tensor count mismatch (313 vs 257 expected by loader).
        # if long_factors:
        #     gguf_writer.add_tensor(f"{gguf_prefix}.rope_factors_long",
        #                            np.array(long_factors, dtype=np.float32))
        #     gguf_writer.add_tensor(f"{gguf_prefix}.rope_factors_short",
        #                            np.array(short_factors, dtype=np.float32))

    gguf_writer.open_output_file(output_path)
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"  → {output_path}")


# ==============================================================================
# Acoustic GGUF writer
# ==============================================================================
def export_acoustic(sf: SafeTensorFile, vae_state_dict: dict, config: dict, output_path: str):
    """Export acoustic components as custom voxcpm2-acoustic GGUF."""
    lm_cfg = config['lm_config']
    encoder_cfg = config.get('encoder_config', {})
    dit_cfg = config.get('dit_config', {})
    vae_cfg = config.get('audio_vae_config', {})

    n_embd = lm_cfg['hidden_size']  # 2048
    n_residual_layers = config.get('residual_lm_num_layers', 8)
    n_encoder_layers = encoder_cfg.get('num_layers', 12)
    n_dit_layers = dit_cfg.get('num_layers', 12)
    patch_size = config.get('patch_size', 4)
    feat_dim = config.get('feat_dim', 64)
    fsq_latent_dim = config.get('scalar_quantization_latent_dim', 512)
    fsq_scale = config.get('scalar_quantization_scale', 9)

    print(f"  Acoustic: ResLM={n_residual_layers}l, LocEnc={n_encoder_layers}l, "
          f"LocDiT={n_dit_layers}l, FSQ dim={fsq_latent_dim}")
    print(f"            patch_size={patch_size}, feat_dim={feat_dim}")

    gguf_writer = GGUFWriter(output_path, "voxcpm2-acoustic")

    # Metadata
    gguf_writer.add_uint32("voxcpm2.patch_size", patch_size)
    gguf_writer.add_uint32("voxcpm2.feat_dim", feat_dim)
    gguf_writer.add_uint32("voxcpm2.residual_lm.n_layer", n_residual_layers)
    gguf_writer.add_uint32("voxcpm2.residual_lm.n_embd", n_embd)
    gguf_writer.add_bool("voxcpm2.residual_lm.no_rope", config.get('residual_lm_no_rope', True))
    gguf_writer.add_uint32("voxcpm2.locenc.n_layer", n_encoder_layers)
    gguf_writer.add_uint32("voxcpm2.locenc.n_embd", encoder_cfg.get('hidden_dim', 1024))
    gguf_writer.add_uint32("voxcpm2.locdit.n_layer", n_dit_layers)
    gguf_writer.add_uint32("voxcpm2.locdit.n_embd", dit_cfg.get('hidden_dim', 1024))
    gguf_writer.add_uint32("voxcpm2.fsq.latent_dim", fsq_latent_dim)
    gguf_writer.add_float32("voxcpm2.fsq.scale", float(fsq_scale))
    gguf_writer.add_float32("voxcpm2.cfm.sigma_min", dit_cfg.get('cfm_config', {}).get('sigma_min', 1e-6))
    gguf_writer.add_float32("voxcpm2.cfm.cfg_rate", dit_cfg.get('cfm_config', {}).get('inference_cfg_rate', 2.0))

    # AudioVAE config
    gguf_writer.add_uint32("voxcpm2.audiovae.encoder_dim", vae_cfg.get('encoder_dim', 128))
    gguf_writer.add_uint32("voxcpm2.audiovae.latent_dim", vae_cfg.get('latent_dim', 64))
    gguf_writer.add_uint32("voxcpm2.audiovae.decoder_dim", vae_cfg.get('decoder_dim', 2048))
    gguf_writer.add_uint32("voxcpm2.audiovae.sample_rate", vae_cfg.get('sample_rate', 16000))
    gguf_writer.add_uint32("voxcpm2.audiovae.out_sample_rate", vae_cfg.get('out_sample_rate', 48000))
    encoder_rates = vae_cfg.get('encoder_rates', [2, 5, 8, 8])
    decoder_rates = vae_cfg.get('decoder_rates', [8, 6, 5, 2, 2, 2])
    gguf_writer.add_array("voxcpm2.audiovae.encoder_rates", list(encoder_rates))
    gguf_writer.add_array("voxcpm2.audiovae.decoder_rates", list(decoder_rates))
    sr_bins = vae_cfg.get('sr_bin_boundaries', [20000, 30000, 40000])
    if sr_bins:
        gguf_writer.add_array("voxcpm2.audiovae.sr_bin_boundaries", list(sr_bins))
    gguf_writer.add_string("voxcpm2.audiovae.cond_type", vae_cfg.get('cond_type', 'scale_bias'))

    # LongRoPE frequency factors for LocEnc/LocDiT (read from lm_config)
    rope_scaling = lm_cfg.get('rope_scaling', None)
    if rope_scaling and rope_scaling.get('type') == 'longrope':
        long_factors = rope_scaling.get('long_factor', None)
        if long_factors:
            gguf_writer.add_array("voxcpm2.rope.long_factor", [float(v) for v in long_factors])

    print("  Writing acoustic tensors...")

    # ---- ResidualLM ----
    gguf_writer.add_tensor("residual_lm.output_norm.weight",
                           to_f32_if_norm("output_norm", sf.get_tensor("residual_lm.norm.weight")))
    for i in range(n_residual_layers):
        pt_prefix = f"residual_lm.layers.{i}"
        gguf_prefix = f"residual_lm.blk.{i}"
        for pt_suffix, gguf_suffix in [
            (".input_layernorm.weight", ".attn_norm.weight"),
            (".self_attn.q_proj.weight", ".attn_q.weight"),
            (".self_attn.k_proj.weight", ".attn_k.weight"),
            (".self_attn.v_proj.weight", ".attn_v.weight"),
            (".self_attn.o_proj.weight", ".attn_output.weight"),
            (".post_attention_layernorm.weight", ".ffn_norm.weight"),
            (".mlp.gate_proj.weight", ".ffn_gate.weight"),
            (".mlp.up_proj.weight", ".ffn_up.weight"),
            (".mlp.down_proj.weight", ".ffn_down.weight"),
        ]:
            tensor = sf.get_tensor(f"{pt_prefix}{pt_suffix}")
            gguf_writer.add_tensor(f"{gguf_prefix}{gguf_suffix}",
                                   to_f32_if_norm(gguf_suffix, tensor))

    # ---- LocEnc ----
    gguf_writer.add_tensor("locenc.norm.weight",
                           to_f32_if_norm("norm", sf.get_tensor("feat_encoder.encoder.norm.weight")))
    gguf_writer.add_tensor("locenc.in_proj.weight",
                           sf.get_tensor("feat_encoder.in_proj.weight"))
    gguf_writer.add_tensor("locenc.in_proj.bias",
                           sf.get_tensor("feat_encoder.in_proj.bias"))
    # CLS token: squeeze [1,1,1,1024] → [1024]
    cls_token = sf.get_tensor("feat_encoder.special_token")
    cls_token = cls_token.reshape(-1)  # → [1024]
    gguf_writer.add_tensor("locenc.cls_token.weight", cls_token)

    for i in range(n_encoder_layers):
        pt_prefix = f"feat_encoder.encoder.layers.{i}"
        gguf_prefix = f"locenc.blk.{i}"
        for pt_suffix, gguf_suffix in [
            (".input_layernorm.weight", ".attn_norm.weight"),
            (".self_attn.q_proj.weight", ".attn_q.weight"),
            (".self_attn.k_proj.weight", ".attn_k.weight"),
            (".self_attn.v_proj.weight", ".attn_v.weight"),
            (".self_attn.o_proj.weight", ".attn_output.weight"),
            (".post_attention_layernorm.weight", ".ffn_norm.weight"),
            (".mlp.gate_proj.weight", ".ffn_gate.weight"),
            (".mlp.up_proj.weight", ".ffn_up.weight"),
            (".mlp.down_proj.weight", ".ffn_down.weight"),
        ]:
            tensor = sf.get_tensor(f"{pt_prefix}{pt_suffix}")
            gguf_writer.add_tensor(f"{gguf_prefix}{gguf_suffix}",
                                   to_f32_if_norm(gguf_suffix, tensor))

    # ---- LocDiT ----
    gguf_writer.add_tensor("locdit.norm.weight",
                           to_f32_if_norm("norm", sf.get_tensor("feat_decoder.estimator.decoder.norm.weight")))
    gguf_writer.add_tensor("locdit.in_proj.weight",
                           sf.get_tensor("feat_decoder.estimator.in_proj.weight"))
    gguf_writer.add_tensor("locdit.in_proj.bias",
                           sf.get_tensor("feat_decoder.estimator.in_proj.bias"))
    gguf_writer.add_tensor("locdit.out_proj.weight",
                           sf.get_tensor("feat_decoder.estimator.out_proj.weight"))
    gguf_writer.add_tensor("locdit.out_proj.bias",
                           sf.get_tensor("feat_decoder.estimator.out_proj.bias"))
    gguf_writer.add_tensor("locdit.cond_proj.weight",
                           sf.get_tensor("feat_decoder.estimator.cond_proj.weight"))
    gguf_writer.add_tensor("locdit.cond_proj.bias",
                           sf.get_tensor("feat_decoder.estimator.cond_proj.bias"))

    # Time MLP
    for mlp_name, gguf_name in [("time_mlp", "time_mlp"), ("delta_time_mlp", "delta_time_mlp")]:
        for layer_n in ["1", "2"]:
            gguf_writer.add_tensor(
                f"locdit.{gguf_name}.linear_{layer_n}.weight",
                sf.get_tensor(f"feat_decoder.estimator.{mlp_name}.linear_{layer_n}.weight"))
            try:
                gguf_writer.add_tensor(
                    f"locdit.{gguf_name}.linear_{layer_n}.bias",
                    sf.get_tensor(f"feat_decoder.estimator.{mlp_name}.linear_{layer_n}.bias"))
            except KeyError:
                pass  # bias may not exist

    for i in range(n_dit_layers):
        pt_prefix = f"feat_decoder.estimator.decoder.layers.{i}"
        gguf_prefix = f"locdit.blk.{i}"
        for pt_suffix, gguf_suffix in [
            (".input_layernorm.weight", ".attn_norm.weight"),
            (".self_attn.q_proj.weight", ".attn_q.weight"),
            (".self_attn.k_proj.weight", ".attn_k.weight"),
            (".self_attn.v_proj.weight", ".attn_v.weight"),
            (".self_attn.o_proj.weight", ".attn_output.weight"),
            (".post_attention_layernorm.weight", ".ffn_norm.weight"),
            (".mlp.gate_proj.weight", ".ffn_gate.weight"),
            (".mlp.up_proj.weight", ".ffn_up.weight"),
            (".mlp.down_proj.weight", ".ffn_down.weight"),
        ]:
            tensor = sf.get_tensor(f"{pt_prefix}{pt_suffix}")
            gguf_writer.add_tensor(f"{gguf_prefix}{gguf_suffix}",
                                   to_f32_if_norm(gguf_suffix, tensor))

    # ---- FSQ ----
    gguf_writer.add_tensor("fsq.in_proj.weight", sf.get_tensor("fsq_layer.in_proj.weight"))
    gguf_writer.add_tensor("fsq.in_proj.bias", sf.get_tensor("fsq_layer.in_proj.bias"))
    gguf_writer.add_tensor("fsq.out_proj.weight", sf.get_tensor("fsq_layer.out_proj.weight"))
    gguf_writer.add_tensor("fsq.out_proj.bias", sf.get_tensor("fsq_layer.out_proj.bias"))

    # ---- Projections ----
    for name in ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"]:
        gguf_writer.add_tensor(f"projections.{name}.weight",
                               sf.get_tensor(f"{name}.weight"))
        gguf_writer.add_tensor(f"projections.{name}.bias",
                               sf.get_tensor(f"{name}.bias"))
    # residual fusion projection
    gguf_writer.add_tensor("projections.res_fusion_proj.weight",
                           sf.get_tensor("fusion_concat_proj.weight"))
    gguf_writer.add_tensor("projections.res_fusion_proj.bias",
                           sf.get_tensor("fusion_concat_proj.bias"))

    # ---- StopPredictor ----
    gguf_writer.add_tensor("stop_predictor.linear1.weight",
                           sf.get_tensor("stop_proj.weight"))
    gguf_writer.add_tensor("stop_predictor.linear1.bias",
                           sf.get_tensor("stop_proj.bias"))
    gguf_writer.add_tensor("stop_predictor.linear2.weight",
                           sf.get_tensor("stop_head.weight"))

    # ---- AudioVAE (from audiovae.pth) ----
    _write_audiovae_weights(gguf_writer, vae_state_dict, vae_cfg)

    gguf_writer.open_output_file(output_path)
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"  → {output_path}")


def _write_audiovae_weights(gguf_writer: GGUFWriter, state_dict: dict, vae_cfg: dict):
    """Write AudioVAE weights, merging weight_norm (weight_g + weight_v) to single tensors."""
    print("  Writing AudioVAE weights (merging weight_norm)...")

    encoder_dim = vae_cfg.get('encoder_dim', 128)
    decoder_dim = vae_cfg.get('decoder_dim', 2048)
    encoder_rates = vae_cfg.get('encoder_rates', [2, 5, 8, 8])
    decoder_rates = vae_cfg.get('decoder_rates', [8, 6, 5, 2, 2, 2])

    # Build weight map: {base_name: {weight_g: tensor, weight_v: tensor, ...}}
    # First collect all weight_norm pairs
    weight_map = {}

    for key, tensor in state_dict.items():
        # tensor from torch.load is a torch.Tensor
        np_tensor = tensor.cpu().numpy()

        # Handle weight_norm: weight_g + weight_v
        if key.endswith('.weight_g'):
            base = key[:-len('.weight_g')]
            weight_map.setdefault(base, {})['weight_g'] = np_tensor
        elif key.endswith('.weight_v'):
            base = key[:-len('.weight_v')]
            weight_map.setdefault(base, {})['weight_v'] = np_tensor
        elif key.endswith('.weight'):
            base = key[:-len('.weight')]
            weight_map.setdefault(base, {})['weight'] = np_tensor
        elif key.endswith('.bias'):
            base = key[:-len('.bias')]
            weight_map.setdefault(base, {})['bias'] = np_tensor
        elif key.endswith('.alpha'):
            base = key[:-len('.alpha')]
            weight_map.setdefault(base, {})['alpha'] = np_tensor
        elif 'metadata' in key or 'kwargs' in key:
            continue  # skip metadata
        else:
            # Unknown key - write as-is
            weight_map.setdefault(key, {})['direct'] = np_tensor

    # Process each weight
    written = set()
    for base_name, parts in sorted(weight_map.items()):
        gguf_name = map_audiovae_key(base_name)

        if 'weight_g' in parts and 'weight_v' in parts:
            # Merge weight_norm
            merged = merge_weight_norm(parts['weight_g'], parts['weight_v'])
            merged = merged.astype(np.float16)
            gguf_writer.add_tensor(gguf_name, merged)
            written.add(base_name)
        elif 'weight' in parts:
            gguf_writer.add_tensor(gguf_name, parts['weight'].astype(np.float16))
            written.add(base_name)
        elif 'direct' in parts:
            gguf_writer.add_tensor(gguf_name, parts['direct'].astype(np.float16))
            written.add(base_name)

        # Bias
        if 'bias' in parts:
            bias_key = base_name + '.bias' if base_name in written else base_name
            gguf_name_bias = map_audiovae_key(base_name) + ".bias"
            gguf_writer.add_tensor(gguf_name_bias, parts['bias'].astype(np.float16))

        # Alpha (Snake activation parameter)
        if 'alpha' in parts:
            alpha_tensor = parts['alpha']
            # Fix alpha shape for ggml broadcast: [C] → [C, 1]
            if alpha_tensor.ndim == 1:
                alpha_tensor = alpha_tensor[:, np.newaxis]
            gguf_name_alpha = map_audiovae_key(base_name) + ".alpha"
            gguf_writer.add_tensor(gguf_name_alpha, alpha_tensor.astype(np.float16))

    print(f"    Wrote {len(written)} AudioVAE weight tensors (incl. merged weight_norm)")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert VoxCPM2 weights to GGUF format for llama.cpp-omni")
    parser.add_argument("--model", required=True,
                        help="Path to model.safetensors")
    parser.add_argument("--vae", required=True,
                        help="Path to audiovae.pth")
    parser.add_argument("--config", required=True,
                        help="Path to config.json")
    parser.add_argument("--output", default="./",
                        help="Output directory for GGUF files (default: .)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"Architecture: {config.get('architecture', 'unknown')}")

    # Load model tensors
    print(f"Loading safetensors: {args.model}")
    sf = SafeTensorFile(args.model)
    print(f"  {len(sf.keys())} tensors")

    # Load AudioVAE
    import torch
    print(f"Loading AudioVAE: {args.vae}")
    vae_data = torch.load(args.vae, map_location='cpu', weights_only=True)
    vae_state_dict = vae_data.get('state_dict', vae_data)

    # Export BaseLM
    print("\n=== Exporting BaseLM ===")
    base_lm_path = os.path.join(args.output, "VoxCPM2-BaseLM-F16.gguf")
    export_baselm(sf, config, base_lm_path)

    # Export Acoustic
    print("\n=== Exporting Acoustic ===")
    acoustic_path = os.path.join(args.output, "VoxCPM2-Acoustic-F16.gguf")
    export_acoustic(sf, vae_state_dict, config, acoustic_path)

    sf.close()

    # Print file sizes
    print("\n=== Done ===")
    for name, path in [("BaseLM", base_lm_path), ("Acoustic", acoustic_path)]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()

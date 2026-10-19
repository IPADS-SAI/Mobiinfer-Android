#include "models.h"

#include "llama-moe-experts.h"
#include "llama-moe-slot.h"

void llama_model_qwen3moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    switch (hparams.n_layer) {
        case 48: type = LLM_TYPE_30B_A3B; break;
        case 94: type = LLM_TYPE_235B_A22B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

    if (n_expert == 0) {
        throw std::runtime_error("n_expert must be > 0 for QWEN3MOE");
    }
    if (n_expert_used == 0) {
        throw std::runtime_error("n_expert_used must be > 0 for QWEN3MOE");
    }

    // ── decode 窗口的物理布局：按 (family, 量化档位) 分组，每组一个 base ──────
    //
    // 与 qwen35moe 使用同一套外置专家 / SSD 槽位机制，细节见那边更长的注释。
    // Qwen3-MoE 与 Qwen3.5-MoE 的差别：没有融合 gate_up、没有 shared expert，
    // 所以这里只处理 gate / up / down 三个 family。
    const int64_t dec_c = moe_decode_window();
    const int64_t k_pf  = moe_prefetch_slots();
    const std::vector<int32_t> layer_c = moe_layer_windows();   // 下标 = MoE 层序号
    const bool dec_enabled = dec_c > 0 && !layer_c.empty();
    auto width_of = [&](int moe_idx) -> int64_t {
        if (!dec_enabled || moe_idx < 0 || moe_idx >= (int) layer_c.size()) {
            return 0;
        }
        return (int64_t) layer_c[(size_t) moe_idx] + k_pf;
    };
    const int64_t prefill_cap = moe_slot_bank_size();
    const bool phase_reuse = moe_phase_reuse();

    const llm_tensor fam_ids[3] = {
        LLM_TENSOR_FFN_GATE_EXPS,
        LLM_TENSOR_FFN_UP_EXPS,
        LLM_TENSOR_FFN_DOWN_EXPS,
    };
    // family -> (bytes_per_expert -> 层号升序)；一个 bank 只属于一个 device。
    using bank_key = std::pair<size_t, ggml_backend_dev_t>;
    std::map<bank_key, std::vector<int>> groups[3];
    if (moe_slot_bank_enabled()) {
        for (int il = 0; il < (int) n_layer; ++il) {
            for (int fi = 0; fi < 3; ++fi) {
                const auto * e = moe_experts_entry(tn(fam_ids[fi], "weight", il).str().c_str());
                if (e) {
                    groups[fi][{e->bytes_per_expert, dev_layer(il)}].push_back(il);
                }
            }
        }
    }
    std::map<std::string, int64_t> group_slots;
    for (int fi = 0; fi < 3; ++fi) {
        for (const auto & kv : groups[fi]) {
            int64_t group_total = 0;
            for (int il : kv.second) {
                group_total += width_of(il);
            }
            group_slots[tn(fam_ids[fi], "weight", kv.second.front()).str()] =
                dec_enabled ? (phase_reuse ? std::max(prefill_cap, group_total)
                                          : prefill_cap + group_total) : prefill_cap;
        }
    }
    auto slots_for = [&](const LLM_TN_IMPL & name) -> int64_t {
        const auto it = group_slots.find(name.str());
        return it == group_slots.end() ? prefill_cap : it->second;
    };

    // 专家张量内容来自 experts（运行时 preadv 写入），与模型文件无关：
    //  - 必须避开 mmap 文件映射包装（否则启用 mmap 时落进只读映射，写不进去）
    //  - gguf 里也可能没有它（strip_experts.py 剥离过），此时按 manifest 合成
    auto create_routed_tensor = [&](const LLM_TN_IMPL & tensor_name,
                                    const std::initializer_list<int64_t> & dims,
                                    int flags) -> ggml_tensor * {
        const bool required = !(flags & TENSOR_NOT_REQUIRED);
        ggml_tensor * tensor = create_tensor_own_alloc(
            ml, tensor_name, dims, flags | TENSOR_NOT_REQUIRED);

        const auto * entry = moe_experts_entry(tensor_name.str().c_str());

        if (!tensor && entry) {
            tensor = create_expert_tensor_from_manifest(ml, tensor_name, dims, entry->type);
        }
        if (!tensor) {
            if (required && moe_slot_bank_enabled()) {
                throw std::runtime_error(format(
                    "routed tensor '%s' is in neither the GGUF nor the MoE experts",
                    tensor_name.str().c_str()));
            }
            return tensor;
        }
        if (!moe_slot_bank_enabled()) {
            return tensor;
        }
        if (!entry) {
            throw std::runtime_error(format(
                "MoE experts is missing routed tensor '%s'",
                tensor_name.str().c_str()));
        }
        if (ggml_nbytes(tensor) != entry->bytes_per_expert * (size_t) n_expert) {
            throw std::runtime_error(format(
                "expert tensor '%s' byte size mismatch: tensor=%zu experts=%zu",
                tensor_name.str().c_str(), ggml_nbytes(tensor),
                entry->bytes_per_expert * (size_t) n_expert));
        }

        ml.virtualize_expert_tensor(
            tensor,
            n_expert,
            slots_for(tensor_name),
            entry->type,
            entry->bytes_per_expert);
        return tensor;
    };

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

        // MoE branch — routed experts go through the external-expert machinery.
        layer.ffn_gate_exps = create_routed_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps   = create_routed_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_routed_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
    }

    // 每组一个物理 base，组内后层是它的 view（RAM = base，而不是每组一个 base × 组内层数）。
    // 名称：prefill 视图沿用原始 GGUF 名；decode 视图加 `.dec` 后缀。
    if (moe_slot_bank_enabled()) {
        struct fam_t {
            ggml_tensor * llama_layer::* pf;
            ggml_tensor * llama_layer::* pf_dec;
        };
        const fam_t fams[3] = {
            { &llama_layer::ffn_gate_exps, &llama_layer::ffn_gate_exps_dec },
            { &llama_layer::ffn_up_exps,   &llama_layer::ffn_up_exps_dec   },
            { &llama_layer::ffn_down_exps, &llama_layer::ffn_down_exps_dec },
        };

        size_t resident = 0;
        int n_win = 0;
        int n_grp = 0;

        for (int fi = 0; fi < 3; ++fi) {
            for (const auto & kv : groups[fi]) {
                const std::vector<int> & members = kv.second;
                if (members.empty()) {
                    continue;
                }
                const int first = members.front();
                ggml_tensor * base = layers[first].*fams[fi].pf;
                if (!base) {
                    continue;
                }
                const auto * e = moe_experts_entry(ggml_get_name(base));
                if (!e || e->bytes_per_expert == 0) {
                    throw std::runtime_error(format(
                        "MoE base '%s' has no usable experts entry", ggml_get_name(base)));
                }
                if (e->bytes_per_expert != kv.first.first) {
                    throw std::runtime_error(format(
                        "MoE group for '%s' has inconsistent expert bytes", ggml_get_name(base)));
                }
                resident += ggml_nbytes(base);
                n_grp++;

                // 组内非组长的张量转成 base 的 view，避免真正分配内存。
                for (size_t j = 1; j < members.size(); ++j) {
                    ggml_tensor * t = layers[members[j]].*fams[fi].pf;
                    if (t) {
                        ml.alias_expert_slot_tensor(t, base);
                    }
                }

                // 组长张量本身是 base，必须换成 view 才能进图。
                for (size_t j = 0; j < members.size(); ++j) {
                    auto & L = layers[members[j]];
                    if (j == 0) {
                        const std::string orig = ggml_get_name(base);
                        ggml_set_name(base, (orig + ".base").c_str());
                        L.*fams[fi].pf = ml.window_expert_slot_tensor(
                            orig.c_str(), base, 0, prefill_cap, e->bytes_per_expert);
                    }
                    n_win++;
                    if (dec_enabled) {
                        const std::string n2 = format("%s.dec", ggml_get_name(L.*fams[fi].pf));
                        int64_t off = phase_reuse ? 0 : prefill_cap;
                        for (size_t k2 = 0; k2 < j; ++k2) {
                            off += width_of(members[k2]);
                        }
                        const int64_t wj = width_of(members[j]);
                        L.*fams[fi].pf_dec = ml.window_expert_slot_tensor(
                            n2.c_str(), base, off, wj, e->bytes_per_expert);
                    }
                }
            }
        }

        if (phase_reuse && dec_enabled) {
            LLAMA_LOG_INFO("%s: phase-reuse enabled: prefill/decode storage overlaps; prefill warming disabled\n", __func__);
        }
        LLAMA_LOG_INFO(
            "%s: MoE bank %d groups (family x quant), %.2f MiB resident; %d prefill windows%s\n",
            __func__, n_grp, resident / 1024.0 / 1024.0, n_win,
            dec_enabled ? format(", %d decode windows (c=%d, 按层宽度 %s)",
                                 n_win, (int) dec_c,
                                 getenv("LLAMA_MOE_LAYER_WINDOWS") ? "on" : "off").c_str()
                        : " (no decode windows)");
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // 前瞻预取：用本层 MoE 输入预测下一层的 top-k，交给 slot 运行时提前读盘。
        ggml_tensor * lookahead_next = nullptr;
        if (moe_slots && moe_slots->lookahead_prefetch_enabled() &&
            ubatch.n_tokens <= (uint32_t) moe_slots->lookahead_prefetch_max_rows() &&
            il + 1 < n_layer && model.layers[il + 1].ffn_gate_inp != nullptr) {
            ggml_tensor * prefetch_router_logits = build_lora_mm(model.layers[il + 1].ffn_gate_inp, cur);
            cb(prefetch_router_logits, "prefetch_router_logits", il);
            lookahead_next = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, prefetch_router_logits, moe_slots->lookahead_prefetch_depth()));
            cb(lookahead_next, "prefetch_topk", il);
            ggml_build_forward_expand(gf, lookahead_next);
        }
        if (moe_slots) {
            moe_slots->set_lookahead_pred(lookahead_next);
        }

        // decode 阶段选每层独立窗口张量，prefill 用共享窗口。
        const auto & L = model.layers[il];
        ggml_tensor * p_up   = moe_decode_phase && L.ffn_up_exps_dec   ? L.ffn_up_exps_dec   : L.ffn_up_exps;
        ggml_tensor * p_gate = moe_decode_phase && L.ffn_gate_exps_dec ? L.ffn_gate_exps_dec : L.ffn_gate_exps;
        ggml_tensor * p_down = moe_decode_phase && L.ffn_down_exps_dec ? L.ffn_down_exps_dec : L.ffn_down_exps;

        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    p_up,
                    p_gate,
                    p_down,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
        cb(moe_out, "ffn_moe_out", il);
        cur = moe_out;

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

"""Quantization-Cascade Speculative Decoding (QCSD) module."""

import time
import torch
from experiments.phi4_cpu_run import (
    NUM_LAYERS, NUM_HEADS, HEAD_DIM, NUM_KV_HEADS, Q_DIM, KV_DIM, INTER,
    ROTARY_DIM, EOS_TOKEN_IDS, apply_rope, rms_norm, silu,
    build_rope_cache, KVHistory, ASDSLKVTracker, forward_layer, WeightStore
)

@torch.inference_mode()
def forward_layer_batch(
    hidden_batch: torch.Tensor,    # (K, hidden)
    layer_idx: int,
    store: WeightStore,
    kv_hist: KVHistory,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    start_pos: int,
) -> torch.Tensor:
    """Batched forward pass for K tokens through one transformer layer.

    Used by QCSD verify phase: loads each weight matrix ONCE and produces
    K outputs via BLAS GEMM instead of K separate GEMV calls.
    Appends all K tokens' KV to the cache with proper causal masking.
    """
    K = hidden_batch.shape[0]

    residual = hidden_batch
    h = rms_norm(hidden_batch, store.get_norm(layer_idx, "input_layernorm"))

    qkv = store.matmul_batch(layer_idx, "qkv_proj", h)

    q = qkv[:, :Q_DIM].view(K, NUM_HEADS, HEAD_DIM)
    k_new = qkv[:, Q_DIM:Q_DIM + KV_DIM].view(K, NUM_KV_HEADS, HEAD_DIM)
    v_new = qkv[:, Q_DIM + KV_DIM:].view(K, NUM_KV_HEADS, HEAD_DIM)

    for i in range(K):
        pos_i = start_pos + i
        cos_p = rope_cos[pos_i:pos_i + 1]
        sin_p = rope_sin[pos_i:pos_i + 1]
        q[i:i + 1] = apply_rope(q[i:i + 1], cos_p, sin_p)
        k_new[i:i + 1] = apply_rope(k_new[i:i + 1], cos_p, sin_p)

    for i in range(K):
        kv_hist.append(layer_idx, k_new[i], v_new[i])

    k_hist, v_hist = kv_hist.get(layer_idx)
    S = k_hist.shape[0]

    expand = NUM_HEADS // NUM_KV_HEADS
    k_full = (k_hist.unsqueeze(2)
                    .expand(-1, -1, expand, -1)
                    .reshape(S, NUM_HEADS, HEAD_DIM)
                    .permute(1, 0, 2)
                    .unsqueeze(0))
    v_full = (v_hist.unsqueeze(2)
                    .expand(-1, -1, expand, -1)
                    .reshape(S, NUM_HEADS, HEAD_DIM)
                    .permute(1, 0, 2)
                    .unsqueeze(0))

    q_attn = q.permute(1, 0, 2).unsqueeze(0)  # (1, heads, K, head_dim)

    # Causal mask: query i at position start_pos+i attends to KV 0..start_pos+i
    # tril(diagonal=d) keeps j <= i+d; we need j <= i + start_pos
    causal = torch.ones(K, S, dtype=torch.bool).tril(diagonal=start_pos)
    attn_mask = torch.where(causal, 0.0, float('-inf'))
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q_attn, k_full, v_full, attn_mask=attn_mask,
    )
    attn_out = attn_out.squeeze(0).permute(1, 0, 2).reshape(K, Q_DIM)

    hidden_batch = residual + store.matmul_batch(layer_idx, "o_proj", attn_out)

    residual = hidden_batch
    h = rms_norm(hidden_batch, store.get_norm(layer_idx, "post_attention_layernorm"))

    gu = store.matmul_batch(layer_idx, "gate_up_proj", h)
    act = silu(gu[:, :INTER]) * gu[:, INTER:]

    hidden_batch = residual + store.matmul_batch(layer_idx, "down_proj", act)
    return hidden_batch


@torch.inference_mode()
def generate_qcsd(
    prompt: str,
    store: WeightStore,
    tokenizer,
    max_new_tokens: int = 50,
    system_prompt: str = "You are a helpful AI assistant.",
    draft_k: int = 7,
) -> str:
    """Generate tokens using Quantization Cascade Speculative Decoding.

    Uses the 2-bit draft bank to speculatively generate K tokens, then
    verifies against the primary (4-bit) model. Accepted tokens are
    produced at the throughput of batch verification.
    """
    print("\n" + "=" * 66)
    print("ASDSL x Phi-4 - QCSD Speculative Decoding")
    print("=" * 66)
    print(f"Prompt : {prompt!r}")
    print(f"Draft K: {draft_k} | Primary: {store.bits}-bit | Draft: {store._draft_bits}-bit")
    print("-" * 66)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )

    max_seq = len(input_ids) + max_new_tokens + draft_k + 64
    rope_cos, rope_sin = build_rope_cache(max_seq, ROTARY_DIM)
    kv_hist = KVHistory(max_seq=max_seq)
    asdsl_tracker = ASDSLKVTracker()

    def run_forward(token_id: int, pos: int, kv: KVHistory,
                    need_logits: bool = True, use_draft: bool = False):
        hidden = store.embed_f16[token_id].float().unsqueeze(0)
        k_new, v_new = [], []
        for i in range(NUM_LAYERS):
            hidden = forward_layer(hidden, i, store, kv, rope_cos, rope_sin,
                                   pos, use_draft=use_draft)
            k_np, v_np = kv.get_last_np(i)
            k_new.append(k_np)
            v_new.append(v_np)
        asdsl_tracker.record_token(k_new, v_new)
        if not need_logits:
            return None
        hidden = rms_norm(hidden, store.final_norm)
        return store.lm_head_matvec(hidden)

    # Prefill with primary model
    print("Prefill: ", end="", flush=True)
    t_prefill_start = time.perf_counter()
    with torch.inference_mode():
        logits = None
        for pos, tid in enumerate(input_ids):
            is_last = (pos == len(input_ids) - 1)
            logits = run_forward(tid, pos, kv_hist, need_logits=is_last, use_draft=False)
    t_prefill = time.perf_counter() - t_prefill_start
    print(f"done ({len(input_ids)} tokens in {t_prefill:.1f}s)")

    # QCSD decode loop
    print("\nAssistant: ", end="", flush=True)
    generated: list[int] = []
    total_draft = 0
    total_accepted = 0
    t_decode_start = time.perf_counter()

    pos = len(input_ids)
    with torch.inference_mode():
        while len(generated) < max_new_tokens:
            current_token = int(logits.argmax())

            if current_token in EOS_TOKEN_IDS:
                generated.append(current_token)
                tok_text = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens([current_token])
                )
                print(tok_text, end="", flush=True)
                break

            # ── DRAFT PHASE ──────────────────────────────────────
            # Run 2-bit model autoregressively for K steps.
            # Snapshot KV first so we can roll back after drafting.
            kv_snap = kv_hist.snapshot()
            draft_start_pos = pos
            draft_tokens = []
            draft_token = current_token

            for k_step in range(draft_k):
                draft_logits = run_forward(
                    draft_token, draft_start_pos + k_step, kv_hist,
                    need_logits=True, use_draft=True,
                )
                next_draft = int(draft_logits.argmax())
                draft_tokens.append(next_draft)
                draft_token = next_draft
                if next_draft in EOS_TOKEN_IDS:
                    break

            total_draft += len(draft_tokens)
            kv_hist.restore(kv_snap)

            # ── VERIFY PHASE (BATCHED) ───────────────────────────
            # Feed [current_token, d_0, d_1, ..., d_{K-2}] through the
            # primary model in a SINGLE batched forward pass.
            # Weight matrices are loaded ONCE and applied to all K tokens
            # via BLAS GEMM — this is the core QCSD speedup.
            verify_tokens = [current_token] + draft_tokens[:-1] if len(draft_tokens) > 1 else [current_token]
            n_verify = len(verify_tokens)

            # Build batched hidden input: (n_verify, hidden_dim)
            hidden_batch = torch.stack([
                store.embed_f16[tid].float() for tid in verify_tokens
            ])

            # Run all layers with batched matmul
            for i in range(NUM_LAYERS):
                hidden_batch = forward_layer_batch(
                    hidden_batch, i, store, kv_hist,
                    rope_cos, rope_sin, draft_start_pos,
                )

            # LM head on all positions — also batched
            hidden_batch = rms_norm(hidden_batch, store.final_norm)
            all_logits = store.lm_head_matmul_batch(hidden_batch)
            # all_logits shape: (n_verify, vocab_size)

            # Record KV for ASDSL tracker
            for vi in range(n_verify):
                k_new_list, v_new_list = [], []
                for layer in range(NUM_LAYERS):
                    cache_idx = kv_hist._len[layer] - n_verify + vi
                    k_new_list.append(kv_hist.k_buf[layer][cache_idx].numpy())
                    v_new_list.append(kv_hist.v_buf[layer][cache_idx].numpy())
                asdsl_tracker.record_token(k_new_list, v_new_list)

            # ── ACCEPT / REJECT ──────────────────────────────────
            # Logits[i] gives the primary model's prediction after seeing
            # verify_tokens[0..i]. Compare logits[i].argmax() vs draft_tokens[i].
            generated.append(current_token)
            tok_text = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens([current_token])
            )
            print(tok_text, end="", flush=True)
            pos += 1

            accepted = []
            correction_tok = None

            for k_idx in range(len(draft_tokens)):
                if k_idx >= n_verify:
                    break
                ref_tok = int(all_logits[k_idx].argmax())
                if ref_tok == draft_tokens[k_idx]:
                    accepted.append(draft_tokens[k_idx])
                else:
                    correction_tok = ref_tok
                    break

            for tok in accepted:
                generated.append(tok)
                tok_text = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens([tok])
                )
                print(tok_text, end="", flush=True)
                if tok in EOS_TOKEN_IDS:
                    break

            total_accepted += len(accepted)

            # Trim KV cache: verify processed n_verify tokens but we only
            # accepted 1 (current) + len(accepted). Roll back the rest.
            n_keep = 1 + len(accepted)
            if n_keep < n_verify:
                trimmed_snap = kv_hist.snapshot()
                for layer in range(NUM_LAYERS):
                    trimmed_snap["lens"][layer] -= (n_verify - n_keep)
                kv_hist.restore(trimmed_snap)

            pos += len(accepted)

            if any(t in EOS_TOKEN_IDS for t in accepted):
                break

            # ── HANDLE CORRECTION ────────────────────────────────────
            if correction_tok is not None:
                # Re-run the primary model for ref_tok so its KV is correctly populated.
                # This single GEMV restores cache correctness.
                logits = run_forward(correction_tok, pos, kv_hist, need_logits=True, use_draft=False)
                generated.append(correction_tok)
                print(tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens([correction_tok])), end="", flush=True)
                total_accepted += 1
                pos += 1
                if correction_tok in EOS_TOKEN_IDS:
                    break
            else:
                # All drafts accepted — reuse the last verify logit (KV is already correct)
                last_idx = min(len(accepted), n_verify - 1)
                logits = all_logits[last_idx]

    t_decode = time.perf_counter() - t_decode_start
    n_tokens = len(generated)
    tps = n_tokens / t_decode if t_decode > 0 else 0
    accept_rate = total_accepted / max(total_draft, 1)

    response_text = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(generated)
    )

    kv_stats = asdsl_tracker.stats()
    print(f"\n\nGenerated : {n_tokens} tokens  |  {tps:.2f} tok/s  |  decode {t_decode:.1f}s")
    print(f"QCSD      : acceptance rate {accept_rate:.1%}  |  "
          f"drafted {total_draft} / accepted {total_accepted}")
    print(f"ASDSL KV  : {kv_stats['tokens']} tokens tracked  "
          f"| {kv_stats['blocks_used']}/{kv_stats['blocks_capacity']} blocks")
    print("=" * 66)

    stats = {
        "acceptance_rate": accept_rate,
        "drafted": total_draft,
        "accepted": total_accepted,
        "tps": tps
    }
    return generated, stats


class QCSDDecoder:
    def __init__(self, draft_model_bits: int = 2, verifier_model_bits: int = 4, gamma: int = 4):
        self.draft_bits = draft_model_bits
        self.verifier_bits = verifier_model_bits
        self.gamma = gamma

    def generate(self, prompt, store, tokenizer, max_new_tokens, input_ids):
        pass

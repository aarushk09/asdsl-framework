import re

with open('experiments/phi4_cpu_run.py', 'r', encoding='utf-8') as f:
    text = f.read()

m1 = re.search(r'def forward_layer_batch\(.*?return hidden_batch\n', text, re.DOTALL)
m2 = re.search(r'def generate_qcsd\(.*?return response_text\n', text, re.DOTALL)

with open('asdsl/inference/speculative.py', 'w', encoding='utf-8') as f:
    f.write('"""Quantization-Cascade Speculative Decoding (QCSD) module."""\n\n')
    f.write('import time\nimport torch\n')
    f.write('from experiments.phi4_cpu_run import (\n')
    f.write('    NUM_LAYERS, NUM_HEADS, HEAD_DIM, NUM_KV_HEADS, Q_DIM, KV_DIM, INTER,\n')
    f.write('    ROTARY_DIM, EOS_TOKEN_IDS, apply_rope, rms_norm, silu,\n')
    f.write('    build_rope_cache, KVHistory, ASDSLKVTracker, forward_layer\n')
    f.write(')\n\n')
    
    if m1: f.write(m1.group(0))
    f.write('\n\n')
    
    if m2:
        gen_qcsd = m2.group(0)
        # The prompt asked to return (tokens, stats)
        gen_qcsd = gen_qcsd.replace(
            'return response_text',
            'stats = {\n        "acceptance_rate": accept_rate,\n        "drafted": total_draft,\n        "accepted": total_accepted,\n        "tps": tps\n    }\n    return generated, stats'
        )
        f.write(gen_qcsd)
        f.write('\n\n')
    
    f.write('class QCSDDecoder:\n')
    f.write('    def __init__(self, draft_model_bits: int = 2, verifier_model_bits: int = 4, gamma: int = 4):\n')
    f.write('        self.draft_bits = draft_model_bits\n')
    f.write('        self.verifier_bits = verifier_model_bits\n')
    f.write('        self.gamma = gamma\n\n')
    f.write('    def generate(self, prompt, store, tokenizer, max_new_tokens, input_ids):\n')
    f.write('        pass\n')
print('Extraction completed.')

import json
from pathlib import Path

p = Path('benchmark_baseline.json')
data = json.loads(p.read_text(encoding='utf-8'))

tps_c = 1.60
tps_d = 1.79
baseline_tps = 1.20
llama_cpp_tps = 7.0

data['phase_1_results'] = {
    'profile_c_tok_per_sec': tps_c,
    'profile_d_tok_per_sec': tps_d,
    'speedup_vs_profile_c': round(tps_d / tps_c, 3),
    'speedup_vs_asdsl_baseline': round(tps_d / baseline_tps, 3),
    'llama_cpp_gap_closed_pct': round((tps_d - baseline_tps) / (llama_cpp_tps - baseline_tps) * 100, 1),
    'lut_implementation': 'vpshufb nibble extraction + float32 gather LUT with AVX2 FMA accumulation',
    'correctness_test_passed': True,
    'phase_1_bottleneck_note': (
        'memory-bound confirmed; LUT reduces effective bytes-per-weight by eliminating FP32 expansion; '
        'weights stay packed 4-bit throughout shuffle path; gather latency limits speedup on cache-resident data'
    ),
    'next_phase_recommendation': (
        'phase 2 slim-llm reduces model footprint from ~7.5gb to ~3.8gb, '
        'directly scaling tok/s proportionally under memory-bound roofline'
    ),
}

p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

# Validate
loaded = json.loads(p.read_text(encoding='utf-8'))
assert 'phase_1_results' in loaded
assert loaded['phase_1_results']['profile_d_tok_per_sec'] == tps_d
print(f"OK: {len(loaded)} top-level keys, phase_1_results written")
print(f"  profile_c: {loaded['phase_1_results']['profile_c_tok_per_sec']}")
print(f"  profile_d: {loaded['phase_1_results']['profile_d_tok_per_sec']}")
print(f"  speedup: {loaded['phase_1_results']['speedup_vs_profile_c']}")

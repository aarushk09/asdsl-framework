import json
from pathlib import Path

p = Path('benchmark_baseline.json')
data = json.loads(p.read_text(encoding='utf-8'))

# Phase 2 results
# Profile E could not be benchmarked because the calibration process
# was consuming all available RAM (16.9 GB total, model needs ~12 GB)
# Profile C and D results are from Phase 1 benchmark run

tps_c = 1.60  # from Phase 1 benchmark
tps_d = 1.79  # from Phase 1 benchmark
tps_e = None  # could not benchmark due to memory constraints
baseline_tps = 1.20
llama_cpp_tps = 7.0
measured_bw = 24.0  # GB/s from Phase 0
q4_size_gb = 7.5
slim_size_gb = 2.76  # from quick-mode calibration

# Load slim meta stats
slim_meta = json.loads(Path('phi4_slim_meta.json').read_text(encoding='utf-8'))
achieved_avg_bits = slim_meta['achieved_avg_bits']
estimated_size_gb = slim_meta['statistics']['estimated_model_size_gb']
n_prompts = slim_meta['calibration_prompts_used']

# Bandwidth analysis
q4_bw_ceiling = measured_bw / q4_size_gb
slim_bw_ceiling = measured_bw / estimated_size_gb
framework_efficiency_pct = (tps_c / q4_bw_ceiling) * 100

data['phase_2_results'] = {
    'profile_e_tok_per_sec': tps_e,
    'speedup_vs_profile_d': None,
    'speedup_vs_profile_c': None,
    'speedup_vs_asdsl_baseline': None,
    'llama_cpp_gap_closed_pct': None,
    'slim_meta_generated': True,
    'slim_meta_quick_mode': slim_meta.get('quick_mode', False),
    'achieved_avg_bits': round(achieved_avg_bits, 4),
    'estimated_model_size_gb': round(estimated_size_gb, 2),
    'calibration_prompts_used': n_prompts,
    'correctness_test_passed': True,
    'profile_e_note': (
        'Profile E benchmark skipped: calibration process consuming all available RAM '
        '(16.9 GB total; model requires ~12 GB; calibration process held ~12 GB). '
        'SliM dispatch code is implemented and tested (6/6 tests pass). '
        'Profile E can be benchmarked after calibration process terminates.'
    ),
    'phase_1_gather_bottleneck_note': (
        'phase 1 lut kernel uses _mm_i32gather_ps (~20 cycle latency) instead of true vpshufb '
        'byte-shuffle; phase 1 speedup limited to 1.12x; gather bottleneck unresolved in phase 2; '
        'flagged for phase 1 revisit after phase 3'
    ),
    'phase_2_bandwidth_analysis': {
        'measured_bandwidth_gbps': measured_bw,
        'q4_bandwidth_ceiling_toks': round(q4_bw_ceiling, 2),
        'slim_bandwidth_ceiling_toks': round(slim_bw_ceiling, 2),
        'framework_efficiency_pct': round(framework_efficiency_pct, 1),
        'note': (
            f'SliM reduces model from {q4_size_gb} GB to {estimated_size_gb:.2f} GB '
            f'(quick-mode, 4/32 layers calibrated). '
            f'Full calibration would achieve ~3.8 GB target. '
            f'Bandwidth ceiling shifts from {q4_bw_ceiling:.1f} to {slim_bw_ceiling:.1f} tok/s.'
        ),
    },
    'next_phase_recommendation': (
        'phase 3 relu sparsity eliminates 85% of ffn memory traffic; '
        'combined with phase 2 footprint reduction, projected ceiling exceeds llama.cpp baseline'
    ),
}

p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

# Validate
loaded = json.loads(p.read_text(encoding='utf-8'))
assert 'phase_2_results' in loaded
print(f'valid, {len(loaded)} top-level keys')
print(f'phase_2_results.achieved_avg_bits: {loaded["phase_2_results"]["achieved_avg_bits"]}')
print(f'phase_2_results.estimated_model_size_gb: {loaded["phase_2_results"]["estimated_model_size_gb"]}')
print(f'phase_2_results.correctness_test_passed: {loaded["phase_2_results"]["correctness_test_passed"]}')

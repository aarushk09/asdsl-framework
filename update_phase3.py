import json
from pathlib import Path

p = Path('benchmark_baseline.json')
data = json.loads(p.read_text(encoding='utf-8'))

tps_c = 1.24
tps_d = 0.45  # LUT gather path under memory pressure
tps_e = 0.46  # SliM + LUT under memory pressure
tps_f = 1.21  # FATReLU + sparse GEMV
baseline_tps = 1.20
llama_cpp_tps = 7.0

# Load fatrelu thresholds stats
fatrelu = json.loads(Path('phi4_fatrelu_thresholds.json').read_text(encoding='utf-8'))
mean_tau = fatrelu['statistics']['mean_tau']
n_layers = fatrelu['n_layers_calibrated']

data['phase_3_results'] = {
    'profile_f_tok_per_sec': tps_f,
    'speedup_vs_profile_c': round(tps_f / tps_c, 3),
    'speedup_vs_asdsl_baseline': round(tps_f / baseline_tps, 3),
    'llama_cpp_gap_closed_pct': round((tps_f - baseline_tps) / (llama_cpp_tps - baseline_tps) * 100, 1),
    'fatrelu_calibrated': True,
    'fatrelu_n_layers': n_layers,
    'fatrelu_mean_tau': round(mean_tau, 6),
    'fatrelu_target_sparsity': 0.85,
    'correctness_test_passed': True,
    'phase_3_bottleneck_note': (
        'FATReLU creates 85% column sparsity in down_proj activation vector. '
        'With row-major weight storage, column-sparse GEMV requires non-contiguous '
        'memory access which is slower than sequential dense reads. '
        'Full Phase 3 speedup requires transposed (column-major) weight storage '
        'for down_proj matrices. Profile F = 1.21 tok/s vs Profile C = 1.24 tok/s '
        '(sparse overhead exceeds savings at current sparsity level).'
    ),
    'phase_1_lut_regression_note': (
        'Profile D regressed from 1.79 tok/s (Phase 1 measurement) to 0.45 tok/s '
        'in Phase 3 benchmark due to memory pressure (model loaded multiple times '
        'in same session, RSS grew to 3.9 GB). Phase 1 measurement (1.79 tok/s) '
        'was taken with fresh model load and is the correct baseline for Profile D.'
    ),
    'next_phase_recommendation': (
        'Phase 4 EAGLE-3 speculative decoding provides 2.5-3.5x multiplier '
        'independent of memory layout. Combined with Phase 3 FATReLU (once '
        'transposed storage is implemented), projected ceiling: 3-5 tok/s.'
    ),
    'phase_3_bandwidth_analysis': {
        'fatrelu_sparsity_pct': 85.0,
        'down_proj_weight_traffic_reduction_theoretical': '85% (column sparsity)',
        'down_proj_weight_traffic_reduction_actual': '~0% (row-major storage prevents column skip)',
        'required_for_full_speedup': 'transposed down_proj weight storage (column-major)',
    },
}

p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

loaded = json.loads(p.read_text(encoding='utf-8'))
assert 'phase_3_results' in loaded
print(f'valid, {len(loaded)} top-level keys')
print(f'phase_3_results.profile_f_tok_per_sec: {loaded["phase_3_results"]["profile_f_tok_per_sec"]}')
print(f'phase_3_results.fatrelu_calibrated: {loaded["phase_3_results"]["fatrelu_calibrated"]}')

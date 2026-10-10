"""Check saved paired outputs and recompute bills; never run a controller."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def check(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(evaluation):
    from dr_research.io import load_scenario, scenario_fingerprint
    from dr_research.physics import validate_plan
    from dr_research.types import Plan

    protocol = json.loads((evaluation / 'protocol.json').read_text(encoding='utf-8'))
    index = json.loads((evaluation / 'completed.json').read_text(encoding='utf-8'))
    check(index['protocol_id'] == protocol['protocol_id'], 'Protocol index differs')
    rows, pairs = [], []
    scenarios = {}
    for key, saved in sorted(index['runs'].items()):
        report_path = evaluation / (key + '.json')
        trace_path = evaluation / (key + '.npz')
        check(file_hash(report_path) == saved['report_sha256'], 'Report hash differs: ' + key)
        check(file_hash(trace_path) == saved['trace_sha256'], 'Trace hash differs: ' + key)
        report = json.loads(report_path.read_text(encoding='utf-8'))
        check(report['protocol_id'] == protocol['protocol_id'], 'Run protocol differs')
        case = key.split('/')[0]
        if case not in scenarios:
            scenarios[case] = load_scenario(evaluation / case / 'scenario.npz')
        s = scenarios[case]
        check(report['scenario_fingerprint'] == scenario_fingerprint(s), 'Scenario differs')
        check(report['executed_steps'] == s.horizon == 336, 'Incomplete weekly result')
        check(report['window_steps'] == protocol['protocol']['settings']['control']['window_steps'], 'Window differs')
        with np.load(trace_path, allow_pickle=False) as z:
            check(np.array_equal(z['decision_t'], np.arange(s.horizon)), 'Time index differs')
            power, delivery, energy = z['power_kw'], z['delivery_kw'], z['energy_kwh']
            check(power.shape == (s.horizon, len(s.network.pump_cop)), 'Power shape differs')
            physical = validate_plan(s, Plan(0, power, delivery, energy))
            check(physical.feasible and report['accepted'], 'Physical audit failed')
            costs = power.sum(axis=1) * s.dt_hours * s.price_yuan_per_kwh
            check(np.allclose(costs, z['electricity_yuan'], rtol=1e-10, atol=1e-6), 'Per-step bill differs')
            check(np.isclose(costs.sum(), report['total_electricity_yuan'], rtol=1e-10, atol=1e-6), 'Total bill differs')
            check(np.isclose(z['control_latency_s'].sum(), report['control_time_s'], rtol=1e-10, atol=1e-6), 'Timing sum differs')
            check(np.allclose(energy[0], s.initial_energy_kwh, rtol=0, atol=1e-6), 'Initial storage differs')
            check(np.allclose(energy[-1], s.terminal_energy_kwh, rtol=0, atol=1e-6), 'Terminal storage differs')
            electricity = float(power.sum() * s.dt_hours)
            heat = float(s.heat_demand_kw.sum() * s.dt_hours)
            buildings = s.heat_demand_kw.shape[1]
            row = dict(run=key, scale=report['scale'], method=report['method'],
                       scenario_fingerprint=report['scenario_fingerprint'], physical_verified=True,
                       steps=s.horizon, dt_hours=s.dt_hours, window_steps=report['window_steps'],
                       electricity_kwh=electricity, electricity_yuan=float(costs.sum()),
                       weighted_price_yuan_per_kwh=float(costs.sum()/electricity),
                       average_electric_kw=electricity/(s.horizon*s.dt_hours),
                       nominal_electric_kw=float(s.network.pump_max_kw.sum()),
                       nominal_heat_kw=float((s.network.pump_max_kw*s.network.pump_cop).sum()),
                       demand_kwh_th=heat, average_heat_kw=heat/(s.horizon*s.dt_hours),
                       peak_heat_kw=float(s.heat_demand_kw.sum(1).max()), buildings=buildings,
                       average_heat_per_building_kw=heat/(s.horizon*s.dt_hours*buildings),
                       bill_per_building_yuan=float(costs.sum()/buildings),
                       initial_energy_kwh=float(energy[0].sum()), terminal_energy_kwh=float(energy[-1].sum()),
                       control_time_s=report['control_time_s'], latency_p95_s=report['latency_p95_s'],
                       latency_max_s=report['latency_max_s'], fallback_count=report['fallback_count'],
                       deadline_misses=report['deadline_misses'], solver_calls=report['solver_calls'],
                       trace_columns=sorted(z.files))
            if report['method'] == 'gnn_value':
                check(report['solver_calls'] == 0, 'Online GNN called LP')
            rows.append(row)
    for scale in protocol['protocol']['ablation']['evaluated_scales']:
        members = {r['method']: r for r in rows if r['scale'] == scale}
        if set(members) != {'lp_mpc', 'gnn_value'}:
            pairs.append(dict(scale=scale, paired=False, present=sorted(members)))
            continue
        lp, g = members['lp_mpc'], members['gnn_value']
        for field in ('scenario_fingerprint', 'steps', 'dt_hours', 'window_steps', 'trace_columns'):
            check(lp[field] == g[field], 'Paired export mismatch: '+field)
        pairs.append(dict(scale=scale, paired=True,
                          gnn_bill_change_fraction=g['electricity_yuan']/lp['electricity_yuan']-1,
                          lp_to_gnn_time_ratio=lp['control_time_s']/g['control_time_s']))
    return dict(protocol_id=protocol['protocol_id'], completed_runs=len(rows),
                paired_scales=[r['scale'] for r in pairs if r['paired']], pairs=pairs, runs=rows,
                billing_formula='sum_t price_yuan_per_kwh[t] * sum_i power_kw[t,i] * dt_hours',
                scope='Saved complete runs only. Partial progress logs are not weekly results. This audit runs no optimizer or GNN.',
                time_scope='Controller preparation including model loading plus all control-step times; excludes offline training, simulator transition and post-run physical audit.',
                realism='Capacity-scaled synthetic building demand, fixed COP and assumed tariff. No floor-area or measured-load calibration; energy charge only, not a complete utility bill.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.evaluation.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'completed_runs': result['completed_runs'], 'paired_scales': result['paired_scales'], 'output': str(args.out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

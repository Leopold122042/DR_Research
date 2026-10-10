"""Decompose validation candidate coverage and complete-GNN ranking error."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch
from dr_research.config import load_config
from dr_research.experiment_config import load_settings
from dr_research.dataset import load_dataset,unpack_group
from dr_research.training import load_checkpoint
from dr_research.control import construct_reference,solve_control_window
from dr_research.io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--settings',default='configs/comparison.toml');parser.add_argument('--physical-config',default='configs/experiment.toml')
    a=parser.parse_args();p=load_config(a.physical_config);c=load_settings(a.settings);torch.set_num_threads(c['control']['torch_threads'])
    data,manifest=load_dataset(a.dataset,p,c);model,_=load_checkpoint(a.checkpoint,p,c);anchors={};rows=[]
    for row in data['validation']:
        s,state,actions=unpack_group(row)
        if row['case'] not in anchors: anchors[row['case']]=construct_reference(s)
        solved=solve_control_window(s,state,anchors[row['case']],window_steps=c['control']['window_steps'],time_limit_s=c['control']['time_limit_s'],solver_method=c['control']['solver_method'],solver_threads=c['control']['solver_threads'])
        if solved.status!='optimal': raise RuntimeError('Validation continuous optimum unavailable')
        with torch.no_grad():
            prepared=model.prepare(s,state,actions);scores=model.full(prepared)+prepared.current_yuan
        truth=(row['future']+row['current']).numpy();choice=int(scores.argmin());best=float(truth.min())
        coverage=best-float(solved.electricity_yuan);ranking=float(truth[choice]-best)
        if coverage < -1e-4: raise RuntimeError('Fixed-action optimum below continuous optimum')
        rows.append(dict(case=row['case'],t=state.t,candidates=len(actions),n_tanks=s.network.n_tanks,continuous_lp_yuan=float(solved.electricity_yuan),labeled_pool_best_yuan=best,gnn_selected_true_yuan=float(truth[choice]),candidate_coverage_gap_yuan=max(0.,coverage),teacher_ranking_regret_yuan=ranking,coverage_gap_yuan_per_tank=max(0.,coverage)/s.network.n_tanks,ranking_regret_yuan_per_tank=ranking/s.network.n_tanks))
    summary={key:float(np.mean([r[key] for r in rows])) for key in ('coverage_gap_yuan_per_tank','ranking_regret_yuan_per_tank')}
    write_json(a.out,dict(records=rows,summary=summary,dataset_key=manifest['key'],checkpoint_sha256=sha256(Path(a.checkpoint).read_bytes()).hexdigest(),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),scope=f'{len(rows)} stored validation states and their at-most-{max(r["candidates"] for r in rows)} LP-labeled candidates; this decomposition does not measure the full online candidate-pool oracle or attribute all weekly bill differences'))
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()

"""Validation-state ablation of candidate density and message fidelity.

Run after the paired comparison, with no competing benchmark process. This
measures complete-teacher-relative decisions, not weekly economic performance.
Weights stay frozen. Reference states are shared across N and fidelity modes.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from dr_research.config import load_config
from dr_research.experiment_config import load_settings,seed_for
from dr_research.dataset import case_seed
from dr_research.training import load_checkpoint
from dr_research.scenario import generate_scenario
from dr_research.control import construct_reference
from dr_research.types import State,Plan
from dr_research.candidates import generate_candidates
from dr_research.scheduler import evaluate_candidates,calibrate_latency
from dr_research.io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',default='configs/comparison.toml')
    parser.add_argument('--physical-config',default='configs/experiment.toml')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--scales',nargs='+',default=['S1','S5'])
    parser.add_argument('--counts',type=int,nargs='+',default=[64,256,1024])
    parser.add_argument('--steps',type=int,nargs='+',default=[0,60,156,252])
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    p=load_config(args.physical_config);c=load_settings(args.settings);torch.set_num_threads(c['control']['torch_threads'])
    torch.use_deterministic_algorithms(True)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    if any(n<1 or n&(n-1) for n in args.counts) or args.repeats<1: raise ValueError('Invalid density/repeats')
    model,meta=load_checkpoint(args.checkpoint,p,c,'value'); model.eval()
    cases=[];prepared=[]
    week=c['data']['validation_weeks'][0]
    for scale in args.scales:
        s=generate_scenario(p,scale,case_seed(c,'validation',scale,week),weather_week=week);reference=construct_reference(s);cases.append((scale,s,reference))
        pool=generate_candidates(s,s.initial_state(),reference,max(args.counts),seed_for(c['master_seed'],'sobol',s.seed))
        with torch.no_grad(): prepared.append(model.prepare(s,s.initial_state(),pool.actions))
    latency,calibration=calibrate_latency(model,prepared,group_sizes=(1,4,16,c['refinement']['group_size']),repeats=3,warmup=1,seed=seed_for(c['master_seed'],'ablation_latency'))
    write_json(out/'latency_profile.json',latency.to_dict())
    records=[]
    for scale,s,reference in cases:
      for t in args.steps:
        if not 0<=t<s.horizon: raise ValueError('State outside weekly episode')
        state=State(t,reference.energy_kwh[t].copy(),(s.initial_power_kw if t==0 else reference.power_kw[t-1]).copy())
        anchor=Plan(t,reference.power_kw[t:],reference.delivery_kw[t:],reference.energy_kwh[t:])
        previous=None;best=float('inf')
        for count in sorted(set(args.counts)):
            started=perf_counter();pool=generate_candidates(s,state,anchor,count,seed_for(c['master_seed'],'sobol',s.seed));generation=perf_counter()-started
            if previous is not None:
                for k,old in zip(previous.ids,previous.actions):
                    new=pool.candidates[int(np.flatnonzero(pool.ids==k)[0])].action
                    if not np.array_equal(new.power_kw,old.power_kw) or not np.array_equal(new.delivery_kw,old.delivery_kw): raise RuntimeError('Nested sampling failed')
            previous=pool
            with torch.no_grad(): prepared=model.prepare(s,state,pool.actions);truth=(model.full(prepared)+prepared.current_yuan).cpu().numpy()
            minimum=float(truth.min())
            if minimum>best+1e-3: raise RuntimeError('Complete-teacher optimum increased for a nested pool')
            best=minimum
            for repeat in range(args.repeats):
                modes=list(np.random.default_rng(seed_for(c['master_seed'],'ablation_order',scale,t,count,repeat)).permutation(['coarse','full','random','adaptive']))
                for mode in modes:
                    scores,report=evaluate_candidates(model,s,state,pool.actions,error_predictor=model.error_predictor,latency_model=latency,**{**c['refinement'],'mode':mode},seed=seed_for(c['master_seed'],'ablation_refine',scale,t,count,repeat))
                    chosen=int(np.argmin(scores));regret=float(truth[chosen]-minimum)
                    row=dict(scale=scale,validation_week=week,t=t,raw_count=count,candidates=len(pool),repeat=repeat,selected_id=int(pool.ids[chosen]),teacher_minimum_yuan=minimum,teacher_relative_regret_yuan=regret,teacher_relative_regret_yuan_per_tank=regret/s.network.n_tanks,candidate_generation_s=generation,**report)
                    records.append(row)
            print(json.dumps(dict(phase='ablation',scale=scale,t=t,raw_count=count,teacher_minimum_yuan=minimum)),flush=True)
    write_json(out/'results.json',dict(records=records,settings=c,scales=args.scales,counts=args.counts,steps=args.steps,repeats=args.repeats,state_source='validation-week feasible constant-inventory reference states; no test-time parameter selection',comparison='same physical state, immutable teacher weights, nested candidate pools; regret measured against complete teacher, not LP optimum',checkpoint_sha256=sha256(Path(args.checkpoint).read_bytes()).hexdigest(),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),calibration=calibration))

if __name__=='__main__': main()

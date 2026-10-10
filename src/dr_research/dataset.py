"""Grouped offline LP labels and independently explored, physically audited states.

LP provides fixed-first-action continuation targets OFFLINE. Online GNN control
never calls this label solver. Current legality does not imply LP extendibility:
nonoptimal/infeasible labels are recorded, not invented or used as zero costs.
"""
from __future__ import annotations
from hashlib import sha256
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .candidates import generate_candidates
from .continuation import try_continuation,prepare_continuation_context
from .control import construct_reference,solve_control_window
from .experiment_config import digest,seed_for
from .io import save_scenario,load_scenario,write_json
from .physics import transition,validate_plan
from .scenario import generate_scenario
from .types import State,Action,Plan

DATASET_VERSION='energy_role_grouped_v1'

def label_summary(records, window_steps):
    """Derived provenance only: unavailable labels are not all infeasible."""
    unavailable = [row for row in records if not row['available']]
    return dict(candidate_labels_attempted=len(records),
                candidate_labels_available=sum(row['available'] for row in records),
                candidate_labels_unavailable=len(unavailable),
                candidate_labels_infeasible=sum(row['status']=='infeasible' for row in unavailable),
                candidate_labels_unknown=sum(row['status']!='infeasible' for row in unavailable),
                label_status_counts=dict(sorted(Counter(row['status'] for row in records).items())),
                teacher_window_steps=int(window_steps),
                teacher=f'{window_steps}-step rolling LP with common reference terminal, identical fixed first action; common tail cost omitted')

def dataset_key(physical,settings):
    from .weather import weather_manifest
    return digest(dict(physical=physical,control=settings['control'],data=settings['data'],master_seed=settings['master_seed'],weather=weather_manifest(),version=DATASET_VERSION))

def case_seed(settings,split,scale,week):
    return seed_for(settings['master_seed'],'topology',split,scale,week)

def unpack_group(row):
    state=State(int(row['t']),row['energy'].numpy().copy(),row['previous_power'].numpy().copy())
    actions=[Action(p.numpy().copy(),f.numpy().copy()) for p,f in zip(row['power'],row['delivery'])]
    return row['scenario'],state,actions

def exploration_step(s,state,anchor,settings,split,pool=None):
    if split not in ('train','validation'): raise ValueError('Only training and validation exploration is permitted')
    if pool is None:
        pool=generate_candidates(s,state,anchor,settings['data']['sampling_count'],seed_for(settings['master_seed'],'sobol',s.seed))
    rng=np.random.default_rng(seed_for(settings['master_seed'],'exploration',split,s.seed,state.t))
    order=rng.permutation(np.arange(1,len(pool.candidates)))[:16]
    context=prepare_continuation_context(s,state,anchor,settings['control']['safety_window_steps'])
    attempts=0
    for k in order:
        attempts+=1
        result=try_continuation(s,state,anchor,pool.candidates[k],settings['control']['safety_window_steps'],context=context)
        if result.accepted:
            return pool,int(k),result.plan,dict(attempts=attempts,status='accepted',reason=result.reason)
    return pool,0,anchor,dict(attempts=attempts,status='backup',reason='no_exploration_certificate')

def prepare_dataset(physical,settings,output,progress=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=False); (output/'scenarios').mkdir()
    started=perf_counter(); ct,d=settings['control'],settings['data']; torch.set_num_threads(ct['torch_threads'])
    payload={'train':[],'validation':[]}; cases=[]; labels=[]; exploration=[]
    for split,weeks in (('train',d['train_weeks']),('validation',d['validation_weeks'])):
      for scale in d['train_scales']:
       for week in weeks:
        seed=case_seed(settings,split,scale,week)
        s=generate_scenario(physical,scale,seed,weather_week=week)
        filename=f'{split}_{scale}_week{week}.npz'; fingerprint=save_scenario(output/'scenarios'/filename,s)
        reference=construct_reference(s); state=s.initial_state(); anchor=reference
        phase=seed_for(settings['master_seed'],'sample_phase',scale,week)%d['sample_stride']
        sample_steps=set(range(phase,s.horizon,d['sample_stride']))|{0,s.horizon-1}
        powers=[];flows=[];energies=[state.energy_kwh.copy()]; case_rows=[]
        for t in range(s.horizon):
            pool=generate_candidates(s,state,anchor,d['sampling_count'],seed_for(settings['master_seed'],'sobol',s.seed))
            if t in sample_steps:
                indices=np.unique(np.linspace(0,len(pool.candidates)-1,min(d['label_candidates'],len(pool.candidates)),dtype=int))
                successes=[];futures=[];currents=[];ids=[]
                for k in indices:
                    candidate=pool.candidates[k]
                    solved=solve_control_window(s,state,reference,window_steps=ct['window_steps'],fixed_first_action=candidate.action,time_limit_s=ct['time_limit_s'],solver_method=ct['solver_method'],solver_threads=ct.get('solver_threads',1))
                    row=dict(split=split,case=filename,t=t,candidate_id=int(candidate.candidate_id),status=solved.status,available=False)
                    if solved.status=='optimal' and solved.plan is not None:
                        current=float(s.dt_hours*s.price_yuan_per_kwh[t]*candidate.action.power_kw.sum())
                        future=float(solved.electricity_yuan-current)
                        if future < -1e-4: raise RuntimeError('Negative nonnegative-tariff continuation label')
                        successes.append(candidate.action);futures.append(max(0.,future));currents.append(current);ids.append(candidate.candidate_id)
                        row.update(available=True,current_yuan=current,future_yuan=max(0.,future),total_yuan=float(solved.electricity_yuan))
                    labels.append(row)
                if successes:
                    payload[split].append(dict(case=filename,t=t,energy=torch.tensor(state.energy_kwh),previous_power=torch.tensor(state.previous_power_kw),power=torch.tensor(np.stack([a.power_kw for a in successes])),delivery=torch.tensor(np.stack([a.delivery_kw for a in successes])),future=torch.tensor(futures,dtype=torch.float64),current=torch.tensor(currents,dtype=torch.float64),candidate_ids=torch.tensor(ids),raw_pool_size=len(pool.candidates)))
            _,chosen,complete,info=exploration_step(s,state,anchor,settings,split,pool)
            action=pool.candidates[chosen].action
            case_rows.append(dict(split=split,case=filename,t=t,chosen_candidate_id=int(pool.candidates[chosen].candidate_id),pool_count=len(pool.candidates),sampled=t in sample_steps,**info))
            state=transition(s,state,action); anchor=Plan(state.t,complete.power_kw[1:],complete.delivery_kw[1:],complete.energy_kwh[1:])
            powers.append(action.power_kw);flows.append(action.delivery_kw);energies.append(state.energy_kwh.copy())
        audit=validate_plan(s,Plan(0,np.asarray(powers),np.asarray(flows),np.asarray(energies)))
        if not audit.feasible: raise RuntimeError(f'Exploration trajectory failed audit: {audit.violations}')
        exploration.extend(case_rows)
        cases.append(dict(split=split,scale=scale,week=week,seed=seed,file=filename,fingerprint=fingerprint,scenario_id=s.scenario_id,exploration_nonbackup=sum(r['chosen_candidate_id']!=0 for r in case_rows),audit_maximum_residuals=audit.max_residuals))
        if progress: progress(dict(phase='dataset',split=split,scale=scale,week=week,groups=len(payload[split]),labels=len(labels),elapsed_s=perf_counter()-started))
    if any(not v for v in payload.values()): raise RuntimeError('Empty training/validation data')
    with (output/'samples.pt').open('xb') as f: torch.save(payload,f)
    write_json(output/'label_status.json',dict(records=labels)); write_json(output/'exploration.json',dict(records=exploration))
    manifest=dict(version=DATASET_VERSION,key=dataset_key(physical,settings),physical=physical,settings=settings,cases=cases,generation_s=perf_counter()-started,counts={k:dict(groups=len(v),labels=sum(len(r['future']) for r in v)) for k,v in payload.items()},samples_sha256=sha256((output/'samples.pt').read_bytes()).hexdigest(),exploration_sha256=sha256((output/'exploration.json').read_bytes()).hexdigest(),**label_summary(labels,ct['window_steps']),exploration='deterministic random exact-action continuation, up to 16 attempts then certified backup; no LP selection',sample_phase='case-dependent deterministic phase plus initial and final steps')
    write_json(output/'manifest.json',manifest)
    return manifest

def load_dataset(path,physical,settings):
    path=Path(path); meta=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if meta['version']!=DATASET_VERSION or meta['key']!=dataset_key(physical,settings): raise ValueError('Dataset contract mismatch')
    for filename,field in (('samples.pt','samples_sha256'),('exploration.json','exploration_sha256')):
        if sha256((path/filename).read_bytes()).hexdigest()!=meta[field]: raise ValueError('Dataset checksum mismatch')
    payload=torch.load(path/'samples.pt',map_location='cpu',weights_only=True)
    if set(payload)!={'train','validation'}: raise ValueError('Invalid data splits')
    scenarios={c['file']:load_scenario(path/'scenarios'/c['file']) for c in meta['cases']}
    for split,rows in payload.items():
        permitted={c['file'] for c in meta['cases'] if c['split']==split}
        for row in rows:
            if row['case'] not in permitted: raise ValueError('Dataset split mismatch')
            row['scenario']=scenarios[row['case']]
    return payload,meta

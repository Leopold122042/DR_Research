"""Validated experiment contract; physical, split and model choices are explicit."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
from pathlib import Path
import tomllib
from .config import SCALES

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def seed_for(master,*keys):
    return int.from_bytes(hashlib.sha256(json.dumps([master,*keys],separators=(',',':')).encode()).digest()[:4],'little')

def validate_settings(c):
    fields={
      'control':{'window_steps','safety_window_steps','time_limit_s','solver_method','solver_threads','torch_threads','candidate_count','max_continuation_attempts'},
      'data':{'train_weeks','validation_weeks','test_weeks','train_scales','test_scales','sample_stride','label_candidates','sampling_count'},
      'training':{'epochs','batch_size','patience','weight_decay','hidden_dims','latent_dims','learning_rates','coarse_widths','coarse_epochs','coarse_warmup_epochs','temperatures','error_epochs'},
      'refinement':{'mode','time_budget_s','shortlist_size','coverage_count','group_size','max_rounds','reserve_s'},
      'evaluation':{'repeats'}}
    if set(c)!={'schema_version','master_seed',*fields} or type(c['schema_version']) is not int or c['schema_version']!=2:
        raise ValueError('Unsupported experiment schema/keys')
    for block,keys in fields.items():
        if not isinstance(c[block],dict) or set(c[block])!=keys: raise ValueError(f'Invalid fields: {block}')
    def integer(v,name,lo=1,hi=100000):
        if type(v) is not int or not lo<=v<=hi: raise ValueError(f'{name}: invalid integer')
    def number(v,name,lo=0):
        if type(v) not in (int,float) or not math.isfinite(v) or v<lo: raise ValueError(f'{name}: invalid number')
    integer(c['master_seed'],'master_seed',0,2**32-1)
    ct,d,tr,r=c['control'],c['data'],c['training'],c['refinement']
    integer(ct['window_steps'],'window',2,336); integer(ct['safety_window_steps'],'safety window',2,ct['window_steps'])
    for k in ('candidate_count',):
        integer(ct[k],k,2,16384)
        if ct[k]&(ct[k]-1): raise ValueError('Candidate raw count must be a power of two')
    for k in ('solver_threads','torch_threads'): integer(ct[k],k,1,64)
    integer(ct['max_continuation_attempts'],'attempts',1,1024)
    number(ct['time_limit_s'],'deadline',1e-6)
    if ct['solver_method'] not in ('highs-ipm','highs-ds','highs'): raise ValueError('Unsupported solver')
    sets=[]
    for k in ('train_weeks','validation_weeks','test_weeks'):
        if not isinstance(d[k],list) or not d[k] or len(d[k])!=len(set(d[k])): raise ValueError('Invalid weeks')
        for v in d[k]: integer(v,k,0,10)
        sets.append(set(d[k]))
    if any(a&b for a,b in itertools.combinations(sets,2)): raise ValueError('Weather splits overlap')
    for k in ('train_scales','test_scales'):
        if not d[k] or len(d[k])!=len(set(d[k])) or any(v not in SCALES for v in d[k]): raise ValueError('Invalid scales')
    integer(d['sample_stride'],'stride',1,336); integer(d['sampling_count'],'sampling count',2,16384)
    if d['sampling_count']&(d['sampling_count']-1): raise ValueError('Sampling count must be power of two')
    integer(d['label_candidates'],'label count',2,d['sampling_count']+1)
    for k in ('epochs','batch_size','patience','coarse_epochs','error_epochs'): integer(tr[k],k)
    integer(tr['coarse_warmup_epochs'],'warmup',0,10000); number(tr['weight_decay'],'weight decay')
    for k in ('hidden_dims','latent_dims','learning_rates','coarse_widths','temperatures'):
        if not tr[k] or len(tr[k])!=len(set(tr[k])): raise ValueError(f'Invalid search list {k}')
        for v in tr[k]:
            if k in ('learning_rates','temperatures'): number(v,k,1e-8)
            else: integer(v,k,1,512)
    if r['mode'] not in ('full','coarse','random','adaptive'): raise ValueError('Invalid GNN mode')
    number(r['time_budget_s'],'refinement budget',1e-6); number(r['reserve_s'],'reserve')
    if r['time_budget_s']>=ct['time_limit_s'] or r['reserve_s']>=r['time_budget_s']: raise ValueError('Invalid nested time budget')
    for k in ('shortlist_size','group_size','max_rounds'): integer(r[k],k)
    integer(r['coverage_count'],'coverage',0,10000); integer(c['evaluation']['repeats'],'repeats',1,100)

def load_settings(path):
    with Path(path).open('rb') as f: c=tomllib.load(f)
    validate_settings(c)
    return c

def search_space(settings):
    t=settings['training']
    return [dict(hidden_dim=h,latent_dim=l,learning_rate=r) for h,l,r in itertools.product(t['hidden_dims'],t['latent_dims'],t['learning_rates'])]

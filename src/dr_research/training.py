"""Complete teacher, equation-24 coarse HPO, residual and real-latency fitting."""
from __future__ import annotations
import copy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .dataset import dataset_key,load_dataset,unpack_group,case_seed
from .experiment_config import digest,seed_for,search_space
from .gnn import CandidateValueGNN
from .scheduler import MessageErrorPredictor,fit_error_predictor,paired_error_examples,calibrate_latency
from .latency import LatencyTree
from .candidates import generate_candidates
from .control import construct_reference
from .io import write_json
from .physics import transition
from .scenario import generate_scenario
from .types import Plan

CHECKPOINT_VERSION='energy_role_full_method_v1'

def _teacher_group(model,row):
    s,state,actions=unpack_group(row)
    p=model.prepare(s,state,actions)
    return p,model.full(p),row['future']

def gap_scale(rows):
    values=[]
    for row in rows:
        total=row['future']+row['current']
        if len(total)>1:
            v=float(total.std(unbiased=False))/row['scenario'].network.n_tanks
            if v>1e-5: values.append(v)
    return max(float(np.median(values)) if values else 1.,.05)

def teacher_loss(model,row,scale):
    prepared,pred,target=_teacher_group(model,row)
    n=prepared.n_tanks
    total=pred+prepared.current_yuan; truth=target+row['current']
    absolute=((pred-target)/(100*n)).square().mean()
    centered=(((total-total.mean())-(truth-truth.mean()))/(scale*n)).square().mean()
    return .1*absolute+centered

def teacher_metrics(model,rows,scale):
    model.eval(); regrets=[]; squared=[]; hits=[]; losses=[]
    with torch.no_grad():
      for row in rows:
        p,pred,target=_teacher_group(model,row)
        total=pred+p.current_yuan;truth=target+row['current'];chosen=int(total.argmin())
        regret=float(truth[chosen]-truth.min())
        regrets.append(regret/p.n_tanks);hits.append(regret<=1e-5)
        squared.extend(((pred-target)/p.n_tanks).square().tolist())
        losses.append(float((((total-total.mean())-(truth-truth.mean()))/(scale*p.n_tanks)).square().mean()))
    return dict(mean_regret_yuan_per_tank=float(np.mean(regrets)),max_regret_yuan_per_tank=float(max(regrets)),optimal_fraction=float(np.mean(hits)),future_rmse_yuan_per_tank=float(np.sqrt(np.mean(squared))),centered_cost_mse=float(np.mean(losses)))

def fit_model(data,settings,hp,progress=None):
    cfg=settings['training'];torch.set_num_threads(settings['control']['torch_threads'])
    torch.manual_seed(seed_for(settings['master_seed'],'teacher_initialization'))
    torch.use_deterministic_algorithms(True)
    model=CandidateValueGNN(window_steps=settings['control']['window_steps'],hidden_dim=hp['hidden_dim'],latent_dim=hp['latent_dim'],coarse_width=cfg['coarse_widths'][0]).unfreeze_teacher()
    optimizer=torch.optim.AdamW(model.teacher_parameters(),lr=hp['learning_rate'],weight_decay=cfg['weight_decay'])
    scale=gap_scale(data['train']);rng=np.random.default_rng(seed_for(settings['master_seed'],'teacher_order'))
    best=(float('inf'),float('inf'));weights=None;stale=0;history=[];started=perf_counter()
    for epoch in range(cfg['epochs']):
        model.train();order=rng.permutation(len(data['train']));total=0.
        for start in range(0,len(order),cfg['batch_size']):
            batch=order[start:start+cfg['batch_size']];optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss=teacher_loss(model,data['train'][index],scale)/len(batch)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite teacher training')
                loss.backward();total+=float(loss.detach())*len(batch)
            torch.nn.utils.clip_grad_norm_(list(model.teacher_parameters()),5.,error_if_nonfinite=True);optimizer.step()
        metrics=teacher_metrics(model,data['validation'],scale)
        history.append(dict(epoch=epoch+1,training_loss=total/len(order),**metrics))
        key=(metrics['mean_regret_yuan_per_tank'],metrics['centered_cost_mse'])
        if key<best: best=key;weights=copy.deepcopy(model.state_dict());best_epoch=epoch+1;stale=0
        else: stale+=1
        if progress and (epoch==0 or (epoch+1)%10==0): progress(dict(phase='teacher_epoch',hyperparameters=hp,epoch=epoch+1,**metrics))
        if stale>=cfg['patience']: break
    model.load_state_dict(weights);model.eval()
    return model,dict(training_s=perf_counter()-started,best_epoch=best_epoch,history=history,gap_scale_yuan_per_tank=scale,validation=teacher_metrics(model,data['validation'],scale),parameter_count=sum(p.numel() for p in model.parameters()))

def dense_examples(model,rows,settings,*,stride=1,raw_count=None):
    """Teacher-only dense pools regenerated from the stored legal backup action.

    A one-step holder supplies the sampling witness, never a future certificate.
    The complete teacher, not an LP, labels every dense pool for Eq.24.
    """
    examples=[]
    for row in rows[::stride]:
        if not len(row['candidate_ids']) or int(row['candidate_ids'][0])!=0: continue
        s,state,actions=unpack_group(row);backup=actions[0];nxt=transition(s,state,backup)
        holder=Plan(state.t,backup.power_kw[None],backup.delivery_kw[None],np.stack([state.energy_kwh,nxt.energy_kwh]))
        count=raw_count or settings['control']['candidate_count']
        pool=generate_candidates(s,state,holder,count,seed_for(settings['master_seed'],'sobol',s.seed))
        with torch.no_grad():
            p=model.prepare(s,state,pool.actions);teacher=model.full(p)+p.current_yuan
        examples.append(dict(prepared=p,teacher=teacher.detach(),scenario=s,state=state,actions=pool.actions))
    if not examples: raise RuntimeError('No dense teacher pools with a legal backup')
    return examples

def decision_loss(scores,teacher,temperature,scale):
    """Eq.24: expected COMPLETE-GNN regret; teacher differences are detached."""
    regret=((teacher-teacher.min())/scale).detach()
    probability=torch.softmax(-scores/(temperature*scale),dim=0)
    return (probability*regret).sum()

def coarse_metrics(model,examples):
    regrets=[];agreement=[]
    with torch.no_grad():
      for row in examples:
        p=row['prepared'];p.coarse_public=None
        score=model.coarse(p)+p.current_yuan;t=row['teacher'];choice=int(score.argmin())
        regret=float(t[choice]-t.min());regrets.append(regret/p.n_tanks);agreement.append(regret<=1e-5)
    return dict(mean_teacher_regret_yuan_per_tank=float(np.mean(regrets)),max_teacher_regret_yuan_per_tank=float(max(regrets)),teacher_optimal_fraction=float(np.mean(agreement)))

def fit_coarse(teacher,data,settings,output,progress=None):
    cfg=settings['training'];rows=[];best=None;selected=None
    # Dense independent-state supervision, including online density. No LP label
    # or frozen prepared graph is reused for the trainable coarse projections.
    train=dense_examples(teacher,data['train'],settings,stride=2)
    validation=dense_examples(teacher,data['validation'],settings,stride=1)
    scales=[float(r['teacher'].std(unbiased=False))/r['prepared'].n_tanks for r in train if len(r['teacher'])>1]
    fixed_scale=max(float(np.median(scales)) if scales else .05,.05)
    teacher_weights={k:v for k,v in teacher.state_dict().items() if not k.startswith('coarse_branch.')}
    for width in cfg['coarse_widths']:
      for temperature in cfg['temperatures']:
        torch.manual_seed(seed_for(settings['master_seed'],'coarse',width,temperature))
        model=CandidateValueGNN(**{**teacher.architecture,'coarse_width':width})
        model.load_state_dict(teacher_weights,strict=False);model.freeze_teacher()
        optimizer=torch.optim.AdamW(model.coarse_parameters(),lr=.002,weight_decay=cfg['weight_decay'])
        rng=np.random.default_rng(seed_for(settings['master_seed'],'coarse_order'));history=[];trial_best=float('inf');trial_state=None;started=perf_counter()
        for epoch in range(cfg['coarse_warmup_epochs']+cfg['coarse_epochs']):
            losses=[]
            for idx in rng.permutation(len(train)):
                row=train[idx];full=row['prepared'];count=min(full.candidate_count,[17,65,257][(epoch+int(idx))%3])
                # Fresh prepare when using a density subset; public teacher
                # encoders are frozen, so only coarse parameters receive grads.
                with torch.no_grad(): p=model.prepare(row['scenario'],row['state'],row['actions'][:count]);target=model.full(p)+p.current_yuan
                optimizer.zero_grad(set_to_none=True);p.coarse_public=None
                if epoch<cfg['coarse_warmup_epochs']:
                    with torch.no_grad(): h=model.complete_messages(p)
                    loss=(model.coarse_messages(p)-h).square().mean()
                else:
                    fraction=(0.,.125,.5)[(epoch+int(idx))%3]
                    positions=torch.tensor(rng.choice(p.positions,int(p.positions*fraction),replace=False),dtype=torch.long)
                    score=model.mixed(p,positions)+p.current_yuan
                    scale=fixed_scale*p.n_tanks
                    loss=decision_loss(score,target,temperature,scale)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite coarse loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(list(model.coarse_parameters()),5.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            metrics=coarse_metrics(model,validation);history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),**metrics))
            if epoch>=cfg['coarse_warmup_epochs'] and metrics['mean_teacher_regret_yuan_per_tank']<trial_best:
                trial_best=metrics['mean_teacher_regret_yuan_per_tank'];trial_state=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            if progress and (epoch+1)%10==0: progress(dict(phase='coarse_epoch',width=width,temperature=temperature,epoch=epoch+1,**metrics))
        model.load_state_dict(trial_state);report=dict(width=width,temperature=temperature,best_epoch=best_epoch,training_s=perf_counter()-started,history=history,validation=coarse_metrics(model,validation))
        rows.append(report)
        if best is None or trial_best<best: best=trial_best;selected=copy.deepcopy(model)
        if progress: progress(dict(phase='coarse_trial',width=width,temperature=temperature,validation=report['validation']))
    write_json(Path(output)/'coarse_hpo.json',dict(trials=rows,selection='held-out hard decision regret relative to frozen complete GNN',fixed_training_scale_yuan_per_tank=fixed_scale,scale_definition='median training-pool std(complete total bill / n_tanks), frozen across densities and states',dense_train_states=len(train),dense_validation_states=len(validation),candidate_densities=[17,65,settings['control']['candidate_count']+1]))
    return selected,rows,train,validation

def save_checkpoint(path,model,physical,settings,manifest,hp,fit_report,error_predictor=None,latency_model=None):
    meta=dict(version=CHECKPOINT_VERSION,kind='value',trained=True,architecture=model.architecture,physical_digest=digest(physical),control_digest=digest(settings['control']),dataset_key=manifest['key'],dataset_samples_sha256=manifest['samples_sha256'],data_split=settings['data'],master_seed=settings['master_seed'],hyperparameters=hp,fit_report=fit_report,loss='H: group-centered total-cost plus future-cost regression; L: frozen-H relative decision regret (24)',torch_version=str(torch.__version__),error_architecture=error_predictor.architecture if error_predictor is not None else None)
    payload=dict(metadata=meta,state_dict=model.state_dict(),error_state=error_predictor.state_dict() if error_predictor is not None else None,latency=latency_model.to_dict() if latency_model is not None else None)
    with Path(path).open('xb') as f: torch.save(payload,f)
    return meta

def load_checkpoint(path,physical,settings,expected_kind=None):
    payload=torch.load(path,map_location='cpu',weights_only=True);meta=payload['metadata']
    if meta.get('version')!=CHECKPOINT_VERSION or not meta.get('trained') or meta.get('kind')!='value' or expected_kind not in (None,'value'): raise ValueError('Unsupported/untrained checkpoint')
    required={'physical_digest','control_digest','master_seed','data_split','dataset_key','architecture','error_architecture'}
    if not required <= set(meta): raise ValueError('Incomplete checkpoint contract')
    architecture=meta['architecture']
    if (not isinstance(architecture,dict) or set(architecture)!={'window_steps','hidden_dim','latent_dim','coarse_width'}
        or any(type(v) is not int or v<1 for v in architecture.values())
        or architecture['window_steps']!=settings['control']['window_steps']): raise ValueError('Checkpoint architecture/window contract mismatch')
    if meta['physical_digest']!=digest(physical) or meta['control_digest']!=digest(settings['control']): raise ValueError('Checkpoint physical/control contract mismatch')
    if meta['master_seed']!=settings['master_seed'] or meta['data_split']!=settings['data'] or meta['dataset_key']!=dataset_key(physical,settings): raise ValueError('Checkpoint data/seed contract mismatch')
    model=CandidateValueGNN(**meta['architecture']);model.load_state_dict(payload['state_dict'],strict=True);model.eval();model.freeze_teacher()
    if payload['error_state'] is not None:
        error=MessageErrorPredictor(**meta['error_architecture']);error.load_state_dict(payload['error_state']);error.eval()
    else: error=None
    # Auxiliary models are not part of the shared teacher/coarse state dictionary.
    object.__setattr__(model,'error_predictor',error)
    object.__setattr__(model,'latency_model',LatencyTree.from_dict(payload['latency']) if payload['latency'] is not None else None)
    return model,meta

def tune_models(physical,settings,dataset_path,output,kinds=('value',),progress=None,*,resume=False):
    if tuple(kinds)!=('value',): raise ValueError('Only candidate-value GNN is supported')
    torch.set_num_threads(settings['control']['torch_threads'])
    torch.use_deterministic_algorithms(True)
    output=Path(output);started=perf_counter();data,manifest=load_dataset(dataset_path,physical,settings)
    if output.exists():
        if not resume or json.loads((output/'settings.json').read_text(encoding='utf-8'))!=settings:
            raise ValueError('Existing HPO output requires explicit resume with identical settings')
        if (output/'coarse_hpo.json').exists() or (output/'selected.pt').exists():
            raise ValueError('Resume currently supports completed teacher trials before coarse output is committed')
    else:
        output.mkdir(parents=True);write_json(output/'settings.json',settings)
    trials=[];best=None;teacher=None;teacher_report=None;hp_selected=None
    for trial,hp in enumerate(search_space(settings)):
        folder=output/f'trial_{trial:02d}'
        if resume and (folder/'teacher.pt').exists() and (folder/'report.json').exists():
            model,meta=load_checkpoint(folder/'teacher.pt',physical,settings,'value');report=meta['fit_report']
            saved=json.loads((folder/'report.json').read_text(encoding='utf-8'))
            if meta['hyperparameters']!=hp or saved['checkpoint_sha256']!=sha256((folder/'teacher.pt').read_bytes()).hexdigest(): raise ValueError('Teacher resume checksum/hyperparameter mismatch')
        else:
            model,report=fit_model(data,settings,hp,progress)
            folder.mkdir();save_checkpoint(folder/'teacher.pt',model,physical,settings,manifest,hp,report)
        row=dict(trial=trial,hyperparameters=hp,eligible=True,fit=report,checkpoint_sha256=sha256((folder/'teacher.pt').read_bytes()).hexdigest())
        if not (folder/'report.json').exists(): write_json(folder/'report.json',row)
        trials.append(row)
        key=(report['validation']['mean_regret_yuan_per_tank'],report['validation']['centered_cost_mse'])
        if best is None or key<best: best=key;teacher=model;teacher_report=report;hp_selected=hp;best_trial=trial
        if progress: progress(dict(phase='teacher_trial',trial=trial,validation=report['validation']))
    model,coarse_reports,train,validation=fit_coarse(teacher,data,settings,output,progress)
    # Rebuild contexts under the selected frozen teacher; re-fit residuals only
    # AFTER Eq.24 training, with disjoint validation states.
    def pairs(rows):
        prepared=[]
        with torch.no_grad():
            for row in rows: prepared.append(model.prepare(row['scenario'],row['state'],row['actions']))
        return paired_error_examples(model,prepared)
    x,y=pairs(train);vx,vy=pairs(validation)
    error,error_report=fit_error_predictor(x,y,validation_features=vx,validation_norms=vy,epochs=settings['training']['error_epochs'],seed=seed_for(settings['master_seed'],'residual'))
    write_json(output/'error_report.json',error_report)
    calibration=[]
    for scale in settings['data']['test_scales']:
        week=settings['data']['validation_weeks'][0];s=generate_scenario(physical,scale,case_seed(settings,'validation',scale,week),weather_week=week);state=s.initial_state();anchor=construct_reference(s)
        pool=generate_candidates(s,state,anchor,settings['control']['candidate_count'],seed_for(settings['master_seed'],'sobol',s.seed))
        with torch.no_grad(): calibration.append(model.prepare(s,state,pool.actions))
    latency,latency_report=calibrate_latency(model,calibration,group_sizes=(1,4,16,settings['refinement']['group_size']),repeats=3,warmup=1,seed=seed_for(settings['master_seed'],'latency'))
    write_json(output/'latency_report.json',latency_report)
    final_report=dict(teacher=teacher_report,coarse=coarse_reports,error=error_report,latency_holdout={k:v for k,v in latency_report.items() if k.startswith('validation_')},trained_complete=True,trained_coarse=True)
    save_checkpoint(output/'selected.pt',model,physical,settings,manifest,hp_selected,final_report,error,latency)
    # Comparison validates this final six-component artifact, not teacher-only trials.
    trials[best_trial]['checkpoint_sha256']=sha256((output/'selected.pt').read_bytes()).hexdigest()
    result=dict(version=CHECKPOINT_VERSION,dataset_key=manifest['key'],search_space=search_space(settings),models={'value':dict(selected_checkpoint='selected.pt',selected_trial=best_trial,trials=trials)},selection='H by independent-state LP-relative hard regret, L by complete-H-relative hard regret; test week untouched',total_hpo_s=perf_counter()-started,hpo_session_resumed=bool(resume),cumulative_teacher_training_s=sum(r['fit']['training_s'] for r in trials),cumulative_coarse_training_s=sum(r['training_s'] for r in coarse_reports),error_training_s=error_report['training_s'])
    write_json(output/'hpo_report.json',result)
    if progress: progress(dict(phase='training_complete',seconds=result['total_hpo_s'],checkpoint=str(output/'selected.pt')))
    return result

"""Scientific contracts for energy-role complete/coarse/mixed evaluation."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch

from dr_research.config import load_config
from dr_research.control import construct_reference
from dr_research.gnn import CandidateValueGNN, ENERGY_SCALE_KWH
from dr_research.physics import transition
from dr_research.scenario import generate_scenario
from dr_research.types import Action

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def deterministic_torch():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(old)

@pytest.fixture
def case():
    s=generate_scenario(load_config(ROOT/"configs/experiment.toml"),"S1",42,weather_week=0)
    a=construct_reference(s).action(0)
    actions=[a,Action(a.power_kw*.95,a.delivery_kw.copy()),Action(a.power_kw*1.05,a.delivery_kw.copy())]
    model=CandidateValueGNN(window_steps=12,hidden_dim=16,latent_dim=8,coarse_width=2)
    return s,s.initial_state(),actions,model


def test_exact_energy_ledger_and_cost_identity(case):
    s,state,actions,model=case
    p=model.prepare(s,state,actions)
    n=s.network
    charge=p.charging_energy_kwh.numpy()
    delivered=p.delivery_energy_kwh.numpy()
    assert charge.dtype==np.float64 and delivered.dtype==np.float64
    for k,a in enumerate(actions):
        expected=transition(s,state,a)
        np.testing.assert_allclose(p.next_energy_kwh[k].numpy(),expected.energy_kwh,rtol=0,atol=1e-10)
        inventory=np.bincount(n.pump_to_tank,weights=charge[k,:,2],minlength=n.n_tanks)+np.bincount(n.delivery_edges[0],weights=delivered[k,:,2],minlength=n.n_tanks)
        np.testing.assert_allclose(inventory-p.standing_loss_kwh.numpy(),expected.energy_kwh-state.energy_kwh,rtol=0,atol=1e-10)
        assert p.current_yuan[k].item()==pytest.approx(charge[k,:,0].sum()*s.price_yuan_per_kwh[state.t])
        np.testing.assert_allclose(charge[k,:,1],s.dt_hours*(n.pump_cop-1)*a.power_kw)
    assert torch.count_nonzero(p.charging_energy_kwh[:,:,3])==0
    assert torch.count_nonzero(p.delivery_energy_kwh[:,:,:2])==0


def test_complete_projection_cache_matches_dense_formula_and_gradients(case):
    s,state,actions,model=case
    p=model.prepare(s,state,actions)
    assert not p.full_projection_cache_ready
    positions=torch.tensor([0,2,2,p.positions-1])
    fast=model.complete_messages(p,positions)
    assert p.full_projection_cache_ready
    dense=model.complete_messages_uncached(p,positions)
    torch.testing.assert_close(fast,dense,rtol=3e-6,atol=3e-7)
    parameter=model.complete_branch.first.weight
    fast_gradient=torch.autograd.grad(fast.square().sum(),parameter,retain_graph=True)[0]
    dense_gradient=torch.autograd.grad(dense.square().sum(),parameter)[0]
    torch.testing.assert_close(fast_gradient,dense_gradient,rtol=3e-5,atol=3e-6)


def test_background_slot_retains_zero_flow_context(case):
    s,state,actions,model=case
    zero=Action(np.zeros(s.network.n_pumps),np.zeros(s.network.n_edges))
    p=model.prepare(s,state,[zero])
    assert torch.count_nonzero(p.charging_energy_kwh)==0
    assert torch.linalg.vector_norm(model.complete_messages(p))>0
    assert torch.linalg.vector_norm(model.coarse_messages(p))>0


def test_incremental_refinement_matches_functional_mixed_and_full(case,monkeypatch):
    s,state,actions,model=case
    with torch.no_grad():
        p=model.prepare(s,state,actions)
        cache=model.start_cache(p)
        assert not p.full_projection_cache_ready
        torch.testing.assert_close(model.score(cache),model.coarse(p))
        physical=cache.aggregates[:,:,:4].clone()
        old_tanks=cache.tank_contributions.clone()
        calls=[]
        original=model.complete_messages
        def tracked(prepared,positions=None):
            assert positions is not None
            calls.append(torch.as_tensor(positions).clone())
            return original(prepared,positions)
        monkeypatch.setattr(model,"complete_messages",tracked)
        positions=torch.tensor([0,1,0,p.n_pumps+3])
        stats=model.refine(cache,positions)
        assert stats["new_messages"]==3
        assert sum(len(x) for x in calls)==3
        assert cache.complete_message_evaluations==3
        expected=model.mixed(p,positions)
        torch.testing.assert_close(model.score(cache),expected,rtol=3e-6,atol=2e-4)
        assert torch.equal(cache.aggregates[:,:,:4],physical)
        assert torch.equal(cache.tank_contributions[2],old_tanks[2])
        assert model.refine(cache,positions)==dict(new_messages=0,receiver_updates=0,candidate_updates=0)
        model.refine(cache,torch.arange(p.positions))
        assert cache.refined.all() and cache.complete_message_evaluations==p.positions
        monkeypatch.setattr(model,"complete_messages",original)
        torch.testing.assert_close(model.score(cache),model.full(p),rtol=3e-6,atol=2e-4)
        torch.testing.assert_close(model.mixed(p,[]),model.coarse(p),rtol=3e-6,atol=2e-4)


def test_sensitivity_matches_message_autograd_and_is_candidate_local(case):
    s,state,actions,model=case
    p=model.prepare(s,state,actions)
    cache=model.start_cache(p)
    got=model.sensitivity(cache,[1])
    y=cache.messages.detach().clone().requires_grad_(True)
    score=model._readout(p,y)
    gradient=torch.autograd.grad(score[1],y)[0]
    expected=torch.linalg.vector_norm(gradient,dim=-1)
    torch.testing.assert_close(got,expected,rtol=2e-6,atol=2e-5)
    assert torch.count_nonzero(got[0])==0 and torch.count_nonzero(got[2])==0
    assert torch.count_nonzero(got[1])>0


def test_training_freezes_teacher_and_full_refinement_has_zero_coarse_gradient(case):
    s,state,actions,model=case
    model.freeze_teacher()
    p=model.prepare(s,state,actions)
    values=model.coarse(p)
    values.square().mean().backward()
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in model.coarse_parameters())
    assert all(v.grad is None and not v.requires_grad for v in model.teacher_parameters())
    model.zero_grad(set_to_none=True)
    p=model.prepare(s,state,actions)
    model.mixed(p,torch.arange(p.positions)).sum().backward()
    assert all(v.grad is None or v.grad.abs().sum()==0 for v in model.coarse_parameters())


def test_terminal_future_cost_exact_zero_at_all_fidelities(case):
    s,state,actions,model=case
    p=model.prepare(s,replace(state,t=s.horizon-1),actions)
    assert p.baseline_yuan.item()==0
    for values in (model.full(p),model.coarse(p),model.mixed(p,[0])):
        assert torch.equal(values,torch.zeros(len(actions)))
    cache=model.start_cache(p)
    model.refine(cache,torch.arange(p.positions))
    assert torch.count_nonzero(model.score(cache))==0
    assert torch.count_nonzero(model.sensitivity(cache,[0,1,2]))==0


def test_window_visibility_and_candidate_independent_baseline(case):
    s,state,actions,model=case
    state=replace(state,t=13)
    p=model.prepare(s,state,actions)
    changes={}
    for name in ("price_yuan_per_kwh","heat_demand_kw","ambient_c","tank_environment_c",
                 "source_limit_kw","power_lower_kw","power_upper_kw","availability"):
        a=getattr(s,name).copy()
        if a.dtype==bool:
            a[:13]=~a[:13]
            a[25:]=~a[25:]
        else:
            a[:13]+=1000
            a[25:]+=1000
        changes[name]=a
    changed=model.prepare(replace(s,**changes),state,actions)
    torch.testing.assert_close(model.full(p),model.full(changed),rtol=0,atol=0)
    one=model.prepare(s,state,actions[:1])
    torch.testing.assert_close(p.baseline_yuan,one.baseline_yuan)
    torch.testing.assert_close(model.full(p)[:1],model.full(one),atol=2e-4,rtol=3e-6)


def test_pump_relabeling_preserves_value(case):
    s,state,actions,model=case
    order=np.random.default_rng(2).permutation(s.network.n_pumps)
    fields=("pump_to_tank","pump_station","pump_cop","pump_max_kw","pump_min_kw","pump_ramp_kw")
    n=replace(s.network,**{name:getattr(s.network,name)[order] for name in fields},
        pump_models=tuple(s.network.pump_models[i] for i in order))
    other=replace(s,network=n,availability=s.availability[:,order],initial_power_kw=s.initial_power_kw[order])
    reordered=replace(state,previous_power_kw=state.previous_power_kw[order])
    swapped=[Action(a.power_kw[order],a.delivery_kw) for a in actions]
    with torch.no_grad():
        left=model.full(model.prepare(s,state,actions))
        right=model.full(model.prepare(other,reordered,swapped))
    torch.testing.assert_close(left,right,atol=2e-4,rtol=3e-6)


def test_equivalent_delivery_allocations_have_identical_scores():
    s=generate_scenario(load_config(ROOT/"configs/experiment.toml"),"S2",42,weather_week=0)
    n=s.network
    edge={(int(t),int(b)):i for i,(t,b) in enumerate(n.delivery_edges.T)}
    pair=None
    for first in range(n.n_tanks):
        for second in range(first+1,n.n_tanks):
            common=[b for b in range(n.n_buildings) if (first,b) in edge and (second,b) in edge]
            if len(common)>=2:
                pair=(first,second,*common[:2])
                break
        if pair is not None:
            break
    assert pair is not None
    ta,tb,ba,bb=pair
    degree=np.bincount(n.delivery_edges[1],minlength=n.n_buildings)
    first_flow=s.heat_demand_kw[0,n.delivery_edges[1]]/degree[n.delivery_edges[1]]
    second_flow=first_flow.copy()
    cycle=[edge[ta,ba],edge[ta,bb],edge[tb,ba],edge[tb,bb]]
    change=.1*min(first_flow[cycle])
    second_flow[cycle]+=np.array([1.,-1.,-1.,1.])*change
    power=construct_reference(s).power_kw[0]
    actions=[Action(power,first_flow),Action(power,second_flow)]
    model=CandidateValueGNN(hidden_dim=16,latent_dim=8,coarse_width=2)
    with torch.no_grad():
        p=model.prepare(s,s.initial_state(),actions)
        torch.testing.assert_close(p.next_energy_kwh[0],p.next_energy_kwh[1],atol=1e-10,rtol=0)
        for value in (model.full(p),model.coarse(p)):
            torch.testing.assert_close(value[:1],value[1:],atol=1e-6,rtol=0)


def test_checkpoint_roundtrip_and_optimizer_free_prediction(case,tmp_path,monkeypatch):
    import scipy.optimize
    s,state,actions,model=case
    def forbidden(*a,**kw):
        raise AssertionError("Value inference must not call an optimizer")
    monkeypatch.setattr(scipy.optimize,"linprog",forbidden)
    path=tmp_path/"model.pt"
    torch.save(dict(architecture=model.architecture,state_dict=model.state_dict()),path)
    loaded=torch.load(path,weights_only=True)
    restored=CandidateValueGNN(**loaded["architecture"])
    restored.load_state_dict(loaded["state_dict"])
    np.testing.assert_array_equal(model.predict_cost(s,state,actions),restored.predict_cost(s,state,actions))
    assert model.training
    assert model.predict_cost(s,state,[]).shape==(0,)


def test_invalid_positions_shapes_and_dimensions_rejected(case):
    s,state,actions,model=case
    with pytest.raises(ValueError,match="positive integer"):
        CandidateValueGNN(latent_dim=0)
    with pytest.raises(ValueError,match="nonempty"):
        model.prepare(s,state,[])
    with pytest.raises(ValueError,match="physical shapes"):
        model.prepare(s,state,[Action(np.zeros(2),actions[0].delivery_kw)])
    p=model.prepare(s,state,actions)
    for bad in ([-1],[p.positions],[.5]):
        with pytest.raises(ValueError):
            model.complete_messages(p,bad)
    with pytest.raises(ValueError,match="span"):
        model.complete_messages(p,torch.ones(2,dtype=torch.bool))


def test_s5_sparse_energy_message_forward():
    s=generate_scenario(load_config(ROOT/"configs/experiment.toml"),"S5",42,weather_week=0)
    a=construct_reference(s).action(0)
    model=CandidateValueGNN(hidden_dim=16,latent_dim=8,coarse_width=2)
    with torch.no_grad():
        p=model.prepare(s,s.initial_state(),[a,a])
        assert p.charging_energy_kwh.shape==(2,2048,4)
        assert p.receiver_index.numel()==2048
        assert model.error_features(p).shape==(2,2048,model.error_feature_dim)
        cache=model.start_cache(p)
        model.refine(cache,[0,2048,4095])
        assert torch.isfinite(model.score(cache)).all()
        assert model.full(p).shape==(2,)

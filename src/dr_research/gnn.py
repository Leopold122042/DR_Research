"""Energy-role messages with complete, coarse and selectively refined evaluation.

All scores are FUTURE electricity costs in yuan. The exact current interval bill
is exposed separately in Prepared.current_yuan. A candidate-independent physical
heat-demand estimate supplies baseline_yuan; signed learned corrections are sums
of 100-yuan tank contributions and a shared global contribution. The final
interval has exactly zero future cost by construction.

Charging messages retain four exact, float64 energy coordinates (kWh), normalized
by fixed 100-kWh role scales. Only their learned coordinates change fidelity.
Delivery is a common linear energy representation. Contexts are fixed within a
call. The model uses one charging-message block, shared tank updates and an
additive readout, so replacement updates only affected tanks and candidate sums.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .types import Action, Scenario, State

ENERGY_SCALE_KWH = 100.0
COST_SCALE_YUAN = 100.0
CONTEXT_WIDTH = 8
PUMP_CONTEXT_WIDTH = 11
TANK_CONTEXT_WIDTH = 12
EDGE_WIDTH = 2
# Role four (useful delivery) is identically zero on charging edges and omitted
# from BOTH learned operators; the exact physical coordinate remains present.
ACTIVE_ROLES = (0, 1, 2, 3)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _tensor(value, device, *, double=False):
    array = np.asarray(value, dtype=np.float64 if double else np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Model inputs must be finite")
    return torch.as_tensor(array.copy(), device=device)


def _group(values, receivers, count):
    result = np.zeros((len(values), count), dtype=np.float64)
    # An edge-order loop avoids K copies of indices and large temporary tensors.
    for edge, receiver in enumerate(receivers):
        result[:, receiver] += values[:, edge]
    return result


@dataclass
class Prepared:
    candidate_count: int
    n_pumps: int
    n_tanks: int
    n_buildings: int
    terminal: bool
    receiver_index: Tensor
    pump_public: Tensor
    tank_public: Tensor
    global_public: Tensor
    pump_dynamic: Tensor
    tank_dynamic: Tensor
    global_dynamic: Tensor
    edge_attributes: Tensor
    charging_energy_kwh: Tensor
    delivery_energy_kwh: Tensor
    tank_charging_energy: Tensor
    tank_delivery_energy: Tensor
    next_energy_kwh: Tensor
    initial_energy_kwh: Tensor
    standing_loss_kwh: Tensor
    current_yuan: Tensor
    baseline_yuan: Tensor
    error_inputs: Tensor
    full_public: tuple[Tensor, Tensor, Tensor] | None = None
    coarse_public: tuple[Tensor, Tensor, Tensor] | None = None

    @property
    def full_projection_cache_ready(self):
        return self.full_public is not None

    @property
    def device(self):
        return self.pump_public.device

    @property
    def positions(self):
        return self.candidate_count * self.n_pumps


@dataclass
class RefinementCache:
    prepared: Prepared
    messages: Tensor
    refined: Tensor
    aggregates: Tensor
    tank_contributions: Tensor
    global_contributions: Tensor
    future_yuan: Tensor
    complete_message_evaluations: int = 0
    receiver_updates: int = 0
    candidate_updates: int = 0


class _CompleteMessages(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        context = PUMP_CONTEXT_WIDTH + TANK_CONTEXT_WIDTH + EDGE_WIDTH
        self.first = nn.Linear(4 + len(ACTIVE_ROLES)*context, hidden_dim)
        self.second = nn.Linear(hidden_dim, latent_dim)


class _CoarseMessages(nn.Module):
    def __init__(self, coarse_width, latent_dim):
        super().__init__()
        width = len(ACTIVE_ROLES)*coarse_width
        self.sender = nn.Linear(PUMP_CONTEXT_WIDTH, width, bias=False)
        self.receiver = nn.Linear(TANK_CONTEXT_WIDTH, width, bias=False)
        self.edge = nn.Linear(EDGE_WIDTH, width)
        self.output = nn.Linear(width, latent_dim, bias=False)


class CandidateValueGNN(nn.Module):
    """Document equations 16--21 with exact incremental replacement (28b).

    ``prepare`` evaluates public context encoders once for the entire candidate
    pool. Training must create a fresh Prepared per update; its differentiable
    context and projection caches belong to one forward/backward evaluation.
    Raw scenarios/states/actions, rather than Prepared, form the dataset.
    """
    error_feature_dim = 12

    def __init__(self, window_steps=48, hidden_dim=32, latent_dim=8,
                 coarse_width=2, message_chunk_size=32768):
        super().__init__()
        self.window_steps = _integer(window_steps, "window_steps")
        self.hidden_dim = _integer(hidden_dim, "hidden_dim")
        self.latent_dim = _integer(latent_dim, "latent_dim")
        self.coarse_width = _integer(coarse_width, "coarse_width")
        self.message_chunk_size = _integer(message_chunk_size, "message_chunk_size")
        w = self.window_steps
        self.global_encoder = nn.Sequential(nn.Linear(6*w+9, CONTEXT_WIDTH), nn.SiLU())
        self.pump_encoder = nn.Sequential(nn.Linear(8+2*w, CONTEXT_WIDTH), nn.SiLU())
        self.tank_encoder = nn.Sequential(nn.Linear(12+3*w, CONTEXT_WIDTH), nn.SiLU())
        self.complete_branch = _CompleteMessages(self.hidden_dim, self.latent_dim)
        self.coarse_branch = _CoarseMessages(self.coarse_width, self.latent_dim)
        self.tank_update = nn.Sequential(
            nn.Linear(TANK_CONTEXT_WIDTH+8+self.latent_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU())
        self.tank_readout = nn.Sequential(
            nn.Linear(self.hidden_dim+TANK_CONTEXT_WIDTH, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, 1))
        self.global_readout = nn.Sequential(
            nn.Linear(CONTEXT_WIDTH+4, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, 1))

    @property
    def architecture(self):
        return dict(window_steps=self.window_steps, hidden_dim=self.hidden_dim,
                    latent_dim=self.latent_dim, coarse_width=self.coarse_width)

    def coarse_parameters(self):
        return self.coarse_branch.parameters()

    def teacher_parameters(self):
        return (p for name, p in self.named_parameters() if not name.startswith("coarse_branch."))

    def freeze_teacher(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("coarse_branch."))
        return self

    def unfreeze_teacher(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(not name.startswith("coarse_branch."))
        return self

    def prepare(self, s: Scenario, state: State, actions: Sequence[Action]) -> Prepared:
        n, w = s.network, self.window_steps
        if isinstance(state.t, (bool, np.bool_)) or not isinstance(state.t, (int, np.integer)) or not 0 <= state.t < s.horizon:
            raise ValueError("state.t must identify an unexecuted interval")
        if not actions:
            raise ValueError("A candidate pool must be nonempty")
        if state.energy_kwh.shape != (n.n_tanks,) or state.previous_power_kw.shape != (n.n_pumps,):
            raise ValueError("State shape does not match the physical network")
        if (not np.isfinite(state.energy_kwh).all() or not np.isfinite(state.previous_power_kw).all()
                or np.any(state.energy_kwh < -1e-5) or np.any(state.energy_kwh > n.tank_capacity_kwh+1e-5)):
            raise ValueError("State inventory must be finite and physically bounded")
        p = np.asarray([a.power_kw for a in actions], dtype=np.float64)
        f = np.asarray([a.delivery_kw for a in actions], dtype=np.float64)
        if p.shape != (len(actions), n.n_pumps) or f.shape != (len(actions), n.n_edges):
            raise ValueError("Candidate actions have invalid physical shapes")
        if not np.isfinite(p).all() or not np.isfinite(f).all():
            raise ValueError("Candidate actions must be finite")
        device = next(self.parameters()).device
        k, h, ns = len(actions), n.n_pumps, n.n_tanks
        length = min(w, s.horizon-state.t)
        def window(a):
            padded = np.zeros((w,)+np.asarray(a).shape[1:], dtype=np.float64)
            padded[:length] = a[state.t:state.t+length]
            return padded
        maxpower, maxheat = n.pump_max_kw.sum(), (n.pump_cop*n.pump_max_kw).sum()
        # Each building's demand is apportioned by its fixed incident capacities
        # solely for context and a candidate-INDEPENDENT baseline, not control.
        bcap = np.bincount(n.delivery_edges[1], weights=n.edge_capacity_kw, minlength=n.n_buildings)
        weights = n.edge_capacity_kw / np.maximum(bcap[n.delivery_edges[1]], 1e-12)
        allocated = window(s.heat_demand_kw)[:, n.delivery_edges[1]]*weights
        tank_demand = _group(allocated, n.delivery_edges[0], ns)
        station_rating = np.bincount(n.pump_station,
            weights=(n.pump_cop-1)*n.pump_max_kw, minlength=n.n_stations)
        source_window = window(s.source_limit_kw)/np.maximum(station_rating[None, :], 1.)
        station_cap = np.bincount(n.tank_station, weights=n.tank_capacity_kwh, minlength=n.n_stations)
        station_energy = np.bincount(n.tank_station, weights=state.energy_kwh, minlength=n.n_stations)
        station_soc = station_energy/np.maximum(station_cap, 1.)
        hour = (state.t*s.dt_hours)%24
        angle, day = 2*np.pi*hour/24, int(state.t*s.dt_hours//24)%7
        valid = np.zeros(w)
        valid[:length] = 1
        common = np.r_[window(s.price_yuan_per_kwh)/1.5, window(s.ambient_c)/30,
            window(s.heat_demand_kw).sum(axis=1)/maxheat, window(s.power_lower_kw)/maxpower,
            window(s.power_upper_kw)/maxpower, valid,
            [s.dt_hours, np.sin(angle), np.cos(angle), (s.horizon-state.t)/s.horizon,
             state.energy_kwh.sum()/n.tank_capacity_kwh.sum(), state.previous_power_kw.sum()/maxpower,
             n.tank_capacity_kwh.sum()/maxheat/24, day/6, float(day>=5)]]
        pump_static = np.column_stack([n.pump_cop/5, n.pump_max_kw/100,
            n.pump_min_kw/n.pump_max_kw, n.pump_ramp_kw/n.pump_max_kw,
            state.previous_power_kw/n.pump_max_kw, station_soc[n.pump_station],
            n.eta_charge[n.pump_to_tank], n.tank_capacity_kwh[n.pump_to_tank]/1500])
        tank_static = np.column_stack([n.tank_capacity_kwh/1500, n.tank_charge_limit_kw/500,
            n.tank_discharge_limit_kw/500, n.tank_ua_kw_per_k/.1,
            n.tank_heat_capacity_kwh_per_k/100, np.full(ns,n.tank_reference_c/100),
            n.eta_charge,n.eta_discharge,state.energy_kwh/n.tank_capacity_kwh,
            s.terminal_energy_kwh/n.tank_capacity_kwh,
            np.bincount(n.pump_to_tank,minlength=ns)/4,
            np.bincount(n.delivery_edges[0],weights=n.edge_capacity_kw,minlength=ns)/1000])
        public_global = self.global_encoder(_tensor(common,device))
        public_pump = self.pump_encoder(_tensor(np.c_[pump_static, window(s.availability).T,
            source_window[:,n.pump_station].T],device)) + public_global
        public_tank = self.tank_encoder(_tensor(np.c_[tank_static,tank_demand.T/500,
            window(s.tank_environment_c).T/30,source_window[:,n.tank_station].T],device)) + public_global
        charge = _group(p*n.pump_cop, n.pump_to_tank, ns)
        delivery = _group(f,n.delivery_edges[0],ns)
        source = _group(p*(n.pump_cop-1),n.pump_station,n.n_stations)
        source_fraction = source / np.maximum(s.source_limit_kw[state.t][None,:],1.)
        temperature = n.tank_reference_c+state.energy_kwh/n.tank_heat_capacity_kwh_per_k
        loss = s.dt_hours*n.tank_ua_kw_per_k*(temperature-s.tank_environment_c[state.t])
        next_energy = state.energy_kwh + s.dt_hours*(charge*n.eta_charge-delivery/n.eta_discharge)-loss
        pump_dynamic = np.stack([p/n.pump_max_kw, source_fraction[:,n.pump_station],
            np.broadcast_to((p.sum(axis=1)/maxpower)[:,None],p.shape)],axis=-1)
        tank_dynamic = np.stack([next_energy/n.tank_capacity_kwh,charge/500,delivery/500,
            source_fraction[:,n.tank_station]],axis=-1)
        global_dynamic = np.stack([p.sum(axis=1)/maxpower,next_energy.sum(axis=1)/n.tank_capacity_kwh.sum(),
            source.sum(axis=1)/max(station_rating.sum(),1.),delivery.sum(axis=1)/maxheat],axis=-1)
        charging = np.stack([s.dt_hours*p,s.dt_hours*p*(n.pump_cop-1),
            s.dt_hours*p*n.pump_cop*n.eta_charge[n.pump_to_tank],np.zeros_like(p)],axis=-1)
        delivered = np.stack([np.zeros_like(f),np.zeros_like(f),
            -s.dt_hours*f/n.eta_discharge[n.delivery_edges[0]],s.dt_hours*f],axis=-1)
        tank_charge = np.stack([_group(charging[:,:,r],n.pump_to_tank,ns) for r in range(4)],axis=-1)
        tank_delivery = np.stack([_group(delivered[:,:,r],n.delivery_edges[0],ns) for r in range(4)],axis=-1)
        # A fixed demand-following electrical estimate. Standing loss uses the
        # current inventory, forecast environment and rated pump-weighted COP.
        local_power = np.bincount(n.pump_to_tank,weights=n.pump_max_kw,minlength=ns)
        local_heat = np.bincount(n.pump_to_tank,weights=n.pump_max_kw*n.pump_cop,minlength=ns)
        local_cop = np.divide(local_heat,local_power,out=np.ones(ns),where=local_power>0)
        future_loss = n.tank_ua_kw_per_k*(temperature[None,:]-window(s.tank_environment_c))
        baseline_heat = tank_demand/n.eta_discharge[None,:]+future_loss
        baseline = s.dt_hours*np.sum(window(s.price_yuan_per_kwh)[1:length,None]*
            baseline_heat[1:length]/(n.eta_charge*local_cop)[None,:])
        receiver = n.pump_to_tank
        error_inputs = np.concatenate([charging/ENERGY_SCALE_KWH,
            (p/n.pump_max_kw)[:,:,None],(next_energy/n.tank_capacity_kwh)[:,receiver,None],
            (charge/np.maximum(n.tank_charge_limit_kw,1.))[:,receiver,None],
            (delivery/np.maximum(n.tank_discharge_limit_kw,1.))[:,receiver,None],
            source_fraction[:,n.pump_station,None],pump_dynamic[:,:,2:3],
            np.broadcast_to((n.pump_cop/5)[None,:,None],(k,h,1)),
            np.broadcast_to((n.tank_capacity_kwh[receiver]/1500)[None,:,None],(k,h,1))],axis=-1)
        return Prepared(k,h,ns,n.n_buildings,state.t==s.horizon-1,
            torch.as_tensor(receiver.copy(),dtype=torch.long,device=device),
            public_pump,public_tank,public_global,_tensor(pump_dynamic,device),_tensor(tank_dynamic,device),
            _tensor(global_dynamic,device),_tensor(np.c_[n.pump_max_kw/100,n.pump_cop*n.pump_max_kw/500],device),
            _tensor(charging,device,double=True),_tensor(delivered,device,double=True),
            _tensor(tank_charge/ENERGY_SCALE_KWH,device),_tensor(tank_delivery/ENERGY_SCALE_KWH,device),
            _tensor(next_energy,device,double=True),_tensor(state.energy_kwh,device,double=True),
            _tensor(loss,device,double=True),_tensor(s.dt_hours*s.price_yuan_per_kwh[state.t]*p.sum(axis=1),device,double=True),
            _tensor(baseline,device,double=True),_tensor(error_inputs,device))

    def _positions(self, prepared, positions):
        if positions is None:
            return torch.arange(prepared.positions,device=prepared.device)
        indices = torch.as_tensor(positions,device=prepared.device)
        if not indices.numel() and indices.dtype != torch.bool:
            return indices.to(torch.long).reshape(-1)
        if indices.dtype == torch.bool:
            if indices.numel()!=prepared.positions:
                raise ValueError("A position mask must span the complete candidate-message pool")
            return indices.reshape(-1).nonzero().squeeze(1)
        if indices.dtype.is_floating_point or indices.dtype.is_complex:
            raise ValueError("Message positions must be integer indices")
        indices = indices.to(torch.long).reshape(-1)
        if indices.numel() and (indices.min()<0 or indices.max()>=prepared.positions):
            raise ValueError("Message position is outside the pool")
        return indices

    def _context(self,p,indices):
        candidate,edge = indices//p.n_pumps,indices%p.n_pumps
        tank = p.receiver_index[edge]
        sender = torch.cat([p.pump_public[edge],p.pump_dynamic[candidate,edge]],-1)
        receiver = torch.cat([p.tank_public[tank],p.tank_dynamic[candidate,tank]],-1)
        return candidate,edge,tank,sender,receiver

    def _full_blocks(self):
        first = self.complete_branch.first
        block = first.weight[:,4:].reshape(self.hidden_dim,len(ACTIVE_ROLES),-1).permute(1,0,2)
        return first.weight[:,:4],block[:,:,:PUMP_CONTEXT_WIDTH],            block[:,:,PUMP_CONTEXT_WIDTH:PUMP_CONTEXT_WIDTH+TANK_CONTEXT_WIDTH],block[:,:,-EDGE_WIDTH:]

    def _full_projections(self,p):
        if p.full_public is None:
            _,u,v,d = self._full_blocks()
            p.full_public = (
                F.linear(p.pump_public,u[:,:,:CONTEXT_WIDTH].reshape(-1,CONTEXT_WIDTH)).reshape(p.n_pumps,4,self.hidden_dim),
                F.linear(p.tank_public,v[:,:,:CONTEXT_WIDTH].reshape(-1,CONTEXT_WIDTH)).reshape(p.n_tanks,4,self.hidden_dim),
                F.linear(p.edge_attributes,d.reshape(-1,EDGE_WIDTH)).reshape(p.n_pumps,4,self.hidden_dim))
        return p.full_public

    def complete_messages(self,p,positions=None):
        """Compute H learned vectors ONLY at requested charging positions."""
        indices = self._positions(p,positions)
        if not indices.numel():
            return p.pump_public.new_empty((0,self.latent_dim))
        u_public,v_public,d_public = self._full_projections(p)
        wz,u,v,_ = self._full_blocks()
        outputs=[]
        for ids in indices.split(self.message_chunk_size):
            k,e=ids//p.n_pumps,ids%p.n_pumps
            t=p.receiver_index[e]
            z=(p.charging_energy_kwh[k,e]/ENERGY_SCALE_KWH).to(p.pump_public.dtype)
            roles=torch.cat([torch.ones_like(z[:,:1]),z[:,:3]],1)
            sender = F.linear(p.pump_dynamic[k,e],u[:,:,CONTEXT_WIDTH:].reshape(-1,3)).reshape(-1,4,self.hidden_dim)
            # Deduplicate receiver-candidate projections within an incremental batch.
            unique,inverse=torch.unique(k*p.n_tanks+t,sorted=True,return_inverse=True)
            receiver=F.linear(p.tank_dynamic.reshape(-1,4)[unique],v[:,:,CONTEXT_WIDTH:].reshape(-1,4)).reshape(-1,4,self.hidden_dim)
            hidden=F.linear(z,wz,self.complete_branch.first.bias)+((u_public[e]+v_public[t]+d_public[e]+sender+receiver[inverse])*roles[:,:,None]).sum(1)
            outputs.append(self.complete_branch.second(F.relu(hidden)))
        return torch.cat(outputs)

    def complete_messages_uncached(self,p,positions=None):
        """Dense equation (18) reference for projection-equivalence verification."""
        indices=self._positions(p,positions)
        outputs=[]
        for ids in indices.split(self.message_chunk_size):
            k,e,_,u,v=self._context(p,ids)
            z=(p.charging_energy_kwh[k,e]/ENERGY_SCALE_KWH).to(u.dtype)
            roles=torch.cat([torch.ones_like(z[:,:1]),z[:,:3]],1)
            context=torch.cat([u,v,p.edge_attributes[e]],1)
            x=torch.cat([z,(roles[:,:,None]*context[:,None,:]).flatten(1)],1)
            outputs.append(self.complete_branch.second(F.relu(self.complete_branch.first(x))))
        return torch.cat(outputs) if outputs else p.pump_public.new_empty((0,self.latent_dim))

    def coarse_messages(self,p,positions=None):
        indices=self._positions(p,positions)
        branch=self.coarse_branch
        if p.coarse_public is None:
            p.coarse_public=(F.linear(p.pump_public,branch.sender.weight[:,:CONTEXT_WIDTH]),
                F.linear(p.tank_public,branch.receiver.weight[:,:CONTEXT_WIDTH]),branch.edge(p.edge_attributes))
        a,b,d=p.coarse_public
        outputs=[]
        for ids in indices.split(self.message_chunk_size):
            k,e=ids//p.n_pumps,ids%p.n_pumps
            t=p.receiver_index[e]
            raw=a[e]+b[t]+d[e]+F.linear(p.pump_dynamic[k,e],branch.sender.weight[:,CONTEXT_WIDTH:])+                F.linear(p.tank_dynamic[k,t],branch.receiver.weight[:,CONTEXT_WIDTH:])
            gates=1+torch.tanh(raw.reshape(-1,4,self.coarse_width))
            z=(p.charging_energy_kwh[k,e]/ENERGY_SCALE_KWH).to(raw.dtype)
            roles=torch.cat([torch.ones_like(z[:,:1]),z[:,:3]],1)
            outputs.append(branch.output((gates*roles[:,:,None]).flatten(1)))
        return torch.cat(outputs) if outputs else p.pump_public.new_empty((0,self.latent_dim))

    def _aggregate(self,p,messages):
        latent=p.pump_public.new_zeros((p.candidate_count,p.n_tanks,self.latent_dim))
        latent=latent.index_add(1,p.receiver_index,messages.reshape(p.candidate_count,p.n_pumps,self.latent_dim))
        return torch.cat([p.tank_charging_energy,latent],-1)

    def _tank_scores(self,p,aggregates,flat_tanks=None):
        if flat_tanks is None:
            flat_tanks=torch.arange(p.candidate_count*p.n_tanks,device=p.device)
        k,t=flat_tanks//p.n_tanks,flat_tanks%p.n_tanks
        context=torch.cat([p.tank_public[t],p.tank_dynamic[k,t]],-1)
        update=self.tank_update(torch.cat([context,aggregates.reshape(-1,4+self.latent_dim)[flat_tanks],p.tank_delivery_energy[k,t]],-1))
        return COST_SCALE_YUAN*self.tank_readout(torch.cat([update,context],-1)).squeeze(-1).double()

    def _global_scores(self,p):
        values=torch.cat([p.global_public.expand(p.candidate_count,-1),p.global_dynamic],-1)
        return COST_SCALE_YUAN*p.n_tanks*self.global_readout(values).squeeze(-1).double()

    def _readout(self,p,messages):
        if p.terminal:
            # Keep zero connected to the computation graph for batch training.
            return messages.reshape(p.candidate_count,-1).sum(1)*0
        aggregates=self._aggregate(p,messages)
        return p.baseline_yuan+self._global_scores(p)+self._tank_scores(p,aggregates).reshape(p.candidate_count,p.n_tanks).sum(1)

    def full(self,prepared):
        return self._readout(prepared,self.complete_messages(prepared))

    def coarse(self,prepared):
        return self._readout(prepared,self.coarse_messages(prepared))

    def mixed(self,prepared,positions):
        ids=torch.unique(self._positions(prepared,positions))
        messages=self.coarse_messages(prepared)
        if ids.numel():
            messages=messages.index_copy(0,ids,self.complete_messages(prepared,ids))
        return self._readout(prepared,messages)

    def forward(self,prepared):
        return self.full(prepared)

    @torch.no_grad()
    def start_cache(self,prepared):
        messages=self.coarse_messages(prepared).reshape(prepared.candidate_count,prepared.n_pumps,self.latent_dim)
        aggregates=self._aggregate(prepared,messages)
        tanks=self._tank_scores(prepared,aggregates).reshape(prepared.candidate_count,prepared.n_tanks)
        global_cost=self._global_scores(prepared)
        future=prepared.baseline_yuan+global_cost+tanks.sum(1)
        if prepared.terminal:
            future=torch.zeros_like(future)
        return RefinementCache(prepared,messages,torch.zeros((prepared.candidate_count,prepared.n_pumps),dtype=torch.bool,device=prepared.device),
            aggregates,tanks,global_cost,future)

    def score(self,cache):
        return cache.future_yuan

    @torch.no_grad()
    def refine(self,cache,positions):
        p=cache.prepared
        ids=torch.unique(self._positions(p,positions))
        ids=ids[~cache.refined.reshape(-1)[ids]]
        if not ids.numel():
            return dict(new_messages=0,receiver_updates=0,candidate_updates=0)
        k,e=ids//p.n_pumps,ids%p.n_pumps
        receivers=k*p.n_tanks+p.receiver_index[e]
        affected,inverse=torch.unique(receivers,sorted=True,return_inverse=True)
        updated=self.complete_messages(p,ids)
        delta=updated-cache.messages.reshape(-1,self.latent_dim)[ids]
        grouped=delta.new_zeros((len(affected),self.latent_dim)).index_add(0,inverse,delta)
        aggregate=cache.aggregates.reshape(-1,4+self.latent_dim)
        aggregate[affected,4:]+=grouped
        cache.messages.reshape(-1,self.latent_dim)[ids]=updated
        old=cache.tank_contributions.reshape(-1)[affected].clone()
        new=self._tank_scores(p,cache.aggregates,affected)
        cache.tank_contributions.reshape(-1)[affected]=new
        candidates=affected//p.n_tanks
        change=cache.future_yuan.new_zeros(p.candidate_count).index_add(0,candidates,new-old)
        cache.future_yuan+=change
        if p.terminal:
            cache.future_yuan.zero_()
        cache.refined.reshape(-1)[ids]=True
        stats=dict(new_messages=len(ids),receiver_updates=len(affected),candidate_updates=len(torch.unique(candidates)))
        cache.complete_message_evaluations+=stats["new_messages"]
        cache.receiver_updates+=stats["receiver_updates"]
        cache.candidate_updates+=stats["candidate_updates"]
        return stats

    def sensitivity(self,cache,candidate_ids):
        p=cache.prepared
        candidates=torch.unique(torch.as_tensor(candidate_ids,dtype=torch.long,device=p.device).reshape(-1))
        if candidates.numel() and (candidates.min()<0 or candidates.max()>=p.candidate_count):
            raise ValueError("Sensitivity candidate is outside the pool")
        result=p.pump_public.new_zeros((p.candidate_count,p.n_pumps))
        if not candidates.numel() or p.terminal:
            return result
        with torch.enable_grad():
            selected=cache.aggregates[candidates].detach().clone().requires_grad_(True)
            tank=torch.arange(p.n_tanks,device=p.device)
            context=torch.cat([p.tank_public.detach().expand(len(candidates),-1,-1),p.tank_dynamic[candidates].detach()],-1)
            update=self.tank_update(torch.cat([context,selected,p.tank_delivery_energy[candidates].detach()],-1))
            scores=COST_SCALE_YUAN*self.tank_readout(torch.cat([update,context],-1)).sum()
            gradient=torch.autograd.grad(scores,selected,create_graph=False)[0][:,:,4:]
            norms=torch.linalg.vector_norm(gradient,dim=-1)
            result[candidates]=norms[:,p.receiver_index]
        return result.detach()

    def error_features(self,prepared_or_cache):
        p=prepared_or_cache.prepared if isinstance(prepared_or_cache,RefinementCache) else prepared_or_cache
        return p.error_inputs

    def predict_cost(self,s,state,actions,*,batch_size=128):
        """Convenience full evaluator returning future YUAN, not normalized labels."""
        _integer(batch_size,"batch_size")
        if not actions:
            return np.empty(0,dtype=np.float64)
        was_training=self.training
        self.eval()
        try:
            with torch.no_grad():
                p=self.prepare(s,state,actions)
                return self.full(p).cpu().numpy().astype(np.float64)
        finally:
            self.train(was_training)

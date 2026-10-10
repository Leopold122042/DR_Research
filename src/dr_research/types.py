"""Shared physical contracts. Arrays use kW, kWh, degrees C, hours and yuan.

This module has no solver, learning framework or file-system dependencies.
H -> S is a fixed function: pump_to_tank[i] is the sole receiver of pump i.
S -> B is an edge list supporting arbitrary many-to-many connections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Network:
    pump_to_tank: IntArray
    delivery_edges: IntArray  # shape (2, E): tank indices, building indices
    pump_station: IntArray
    tank_station: IntArray
    building_station: IntArray
    pump_cop: FloatArray
    pump_max_kw: FloatArray  # electrical input
    pump_min_kw: FloatArray
    pump_ramp_kw: FloatArray  # absolute change per control interval
    tank_capacity_kwh: FloatArray  # usable energy above reference temperature
    tank_charge_limit_kw: FloatArray  # heat before charging efficiency
    tank_discharge_limit_kw: FloatArray  # useful delivered heat
    tank_ua_kw_per_k: FloatArray
    tank_heat_capacity_kwh_per_k: FloatArray
    tank_reference_c: float
    eta_charge: FloatArray
    eta_discharge: FloatArray
    edge_capacity_kw: FloatArray  # useful delivered heat
    pump_models: tuple[str, ...] = ()

    @property
    def n_pumps(self) -> int:
        return len(self.pump_to_tank)

    @property
    def n_tanks(self) -> int:
        return len(self.tank_capacity_kwh)

    @property
    def n_buildings(self) -> int:
        return len(self.building_station)

    @property
    def n_edges(self) -> int:
        return self.delivery_edges.shape[1]

    @property
    def n_stations(self) -> int:
        return int(self.building_station.max()) + 1


@dataclass(frozen=True)
class State:
    t: int
    energy_kwh: FloatArray
    previous_power_kw: FloatArray


@dataclass(frozen=True)
class Action:
    power_kw: FloatArray
    delivery_kw: FloatArray  # aligned with Network.delivery_edges


@dataclass(frozen=True)
class Scenario:
    network: Network
    dt_hours: float
    price_yuan_per_kwh: FloatArray  # (T,)
    heat_demand_kw: FloatArray  # (T, B)
    ambient_c: FloatArray  # (T,)
    tank_environment_c: FloatArray  # (T, S), soil for underground tanks
    source_limit_kw: FloatArray  # (T, station), source heat Q - P
    power_lower_kw: FloatArray  # (T,), global service band
    power_upper_kw: FloatArray  # (T,)
    availability: NDArray[np.bool_]  # (T, H), exogenous fixed operating schedule
    initial_energy_kwh: FloatArray
    initial_power_kw: FloatArray
    terminal_energy_kwh: FloatArray
    scenario_id: str
    seed: int
    config_digest: str

    @property
    def horizon(self) -> int:
        return len(self.price_yuan_per_kwh)

    def initial_state(self) -> State:
        return State(0, self.initial_energy_kwh.copy(), self.initial_power_kw.copy())


@dataclass(frozen=True)
class Plan:
    start_t: int
    power_kw: FloatArray  # (remaining horizon, H)
    delivery_kw: FloatArray  # (remaining horizon, E)
    energy_kwh: FloatArray  # (remaining horizon + 1, S)

    def action(self, offset: int) -> Action:
        return Action(self.power_kw[offset].copy(), self.delivery_kw[offset].copy())


@dataclass(frozen=True)
class ValidationReport:
    feasible: bool
    violations: tuple[str, ...]
    max_residuals: dict[str, float]
    electricity_yuan: float | None = None


@dataclass(frozen=True)
class SolveResult:
    status: str
    plan: Plan | None
    electricity_yuan: float | None
    lower_bound_yuan: float | None
    timings_s: dict[str, float]
    message: str
    statistics: dict[str, Any] = field(default_factory=dict)




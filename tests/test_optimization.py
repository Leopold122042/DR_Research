"""Small analytic dispatch cases test physical meaning rather than matrix copies."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import isspmatrix_csr

from dr_research.optimization import build_linear_program, solve_dispatch
from dr_research.physics import validate_plan
from dr_research.types import Action, Network, Scenario, State, ValidationReport


def tiny_scenario() -> Scenario:
    def vector(value):
        return np.array([value], dtype=float)

    net = Network(
        pump_to_tank=np.array([0], dtype=np.int64),
        delivery_edges=np.array([[0], [0]], dtype=np.int64),
        pump_station=np.array([0], dtype=np.int64),
        tank_station=np.array([0], dtype=np.int64),
        building_station=np.array([0], dtype=np.int64),
        pump_cop=vector(2), pump_max_kw=vector(1), pump_min_kw=vector(0),
        pump_ramp_kw=vector(1), tank_capacity_kwh=vector(2),
        tank_charge_limit_kw=vector(2), tank_discharge_limit_kw=vector(2),
        tank_ua_kw_per_k=vector(0), tank_heat_capacity_kwh_per_k=vector(1),
        tank_reference_c=70., eta_charge=vector(1), eta_discharge=vector(1),
        edge_capacity_kw=vector(2), pump_models=("analytic",),
    )
    return Scenario(net, 1., np.array([1., 3.]), np.ones((2, 1)), np.zeros(2),
                    np.full((2, 1), 10.), np.ones((2, 1)), np.zeros(2), np.ones(2),
                    np.ones((2, 1), dtype=bool), vector(0), vector(0), vector(0),
                    "analytic-two-period", 0, "test")


class DispatchTests(unittest.TestCase):
    def test_default_explicitly_selects_ipm_and_reports_crossover(self):
        with patch("dr_research.optimization.linprog", wraps=linprog) as mocked:
            solved = solve_dispatch(tiny_scenario())
        self.assertEqual(solved.status, "optimal", solved.message)
        self.assertEqual(mocked.call_args.kwargs["method"], "highs-ipm")
        self.assertEqual(solved.statistics["solver_method"], "highs-ipm")
        self.assertIsInstance(solved.statistics["crossover_iterations"], int)
        self.assertGreaterEqual(solved.statistics["crossover_iterations"], 0)

    def test_invalid_solver_method_fails_before_solver(self):
        for method in ("interior-point", "invalid", None, ["highs-ipm"]):
            with self.subTest(method=method):
                with patch("dr_research.optimization.linprog") as mocked:
                    solved = solve_dispatch(tiny_scenario(), solver_method=method)
                self.assertEqual(solved.status, "error")
                self.assertIn("solver_method", solved.message)
                self.assertIsNone(solved.plan)
                mocked.assert_not_called()

    def test_ipm_and_dual_simplex_agree_on_cost_and_feasibility(self):
        scenario = tiny_scenario()
        bills = []
        for method in ("highs-ipm", "highs-ds", "highs"):
            with self.subTest(method=method):
                solved = solve_dispatch(scenario, solver_method=method)
                self.assertEqual(solved.status, "optimal", solved.message)
                self.assertEqual(solved.statistics["solver_method"], method)
                self.assertTrue(validate_plan(scenario, solved.plan).feasible)
                self.assertAlmostEqual(solved.electricity_yuan, 1.)
                bills.append(solved.electricity_yuan)
        np.testing.assert_allclose(bills, bills[0], atol=1e-8)

    def test_two_period_storage_moves_heat_to_cheap_interval(self):
        scenario = tiny_scenario()
        solved = solve_dispatch(scenario)
        self.assertEqual(solved.status, "optimal", solved.message)
        self.assertAlmostEqual(solved.electricity_yuan, 1.)
        self.assertAlmostEqual(solved.lower_bound_yuan, 1.)
        np.testing.assert_allclose(solved.plan.power_kw[:, 0], [1., 0.], atol=1e-8)
        np.testing.assert_allclose(solved.plan.energy_kwh[:, 0], [0., 1., 0.], atol=1e-8)
        self.assertEqual(set(solved.timings_s), {"assembly", "solve", "validation", "total"})

    def test_fixed_first_action_returns_its_complete_bill(self):
        scenario = tiny_scenario()
        action = Action(np.array([.5]), np.array([1.]))
        solved = solve_dispatch(scenario, fixed_first_action=action)
        self.assertEqual(solved.status, "optimal", solved.message)
        self.assertAlmostEqual(solved.electricity_yuan, 2.)
        np.testing.assert_allclose(solved.plan.power_kw[:, 0], [.5, .5])
        impossible = solve_dispatch(scenario, fixed_first_action=Action(np.array([0.]), np.array([1.])))
        self.assertEqual(impossible.status, "infeasible")
        self.assertIsNone(impossible.plan)
        self.assertIsNone(impossible.electricity_yuan)
        self.assertIsNone(impossible.lower_bound_yuan)

    def test_fixed_action_does_not_overwrite_equipment_bounds(self):
        solved = solve_dispatch(tiny_scenario(),
                                fixed_first_action=Action(np.array([2.]), np.array([1.])))
        self.assertEqual(solved.status, "infeasible")
        self.assertGreater(solved.statistics["inconsistent_bound_count"], 0)

    def test_future_state_uses_remaining_prices_and_previous_power(self):
        scenario = tiny_scenario()
        solved = solve_dispatch(scenario, state=State(1, np.array([1.]), np.array([1.])))
        self.assertEqual(solved.status, "optimal", solved.message)
        self.assertEqual(solved.plan.start_t, 1)
        self.assertEqual(solved.plan.power_kw.shape, (1, 1))
        self.assertAlmostEqual(solved.electricity_yuan, 0.)
        np.testing.assert_allclose(solved.plan.energy_kwh[:, 0], [1., 0.])
        done = solve_dispatch(scenario, state=State(2, np.array([0.]), np.array([0.])))
        self.assertEqual(done.status, "optimal", done.message)
        self.assertEqual(done.plan.power_kw.shape, (0, 1))
        self.assertAlmostEqual(done.electricity_yuan, 0.)

    def test_each_physical_capacity_can_make_a_plan_infeasible(self):
        base = tiny_scenario()
        cases = {
            "source": replace(base, source_limit_kw=np.full((2, 1), .1)),
            "charge": replace(base, network=replace(base.network, tank_charge_limit_kw=np.array([.5]))),
            "useful_delivery": replace(base, network=replace(base.network, tank_discharge_limit_kw=np.array([.5]))),
            "edge": replace(base, network=replace(base.network, edge_capacity_kw=np.array([.5]))),
            "ramp": replace(base, network=replace(base.network, pump_ramp_kw=np.array([.1]))),
            "unavailable": replace(base, availability=np.zeros((2, 1), dtype=bool)),
            "global_power_upper": replace(base, power_upper_kw=np.full(2, .1)),
            "global_power_lower": replace(base, power_lower_kw=np.ones(2)),
        }
        for name, scenario in cases.items():
            with self.subTest(name=name):
                solved = solve_dispatch(scenario)
                self.assertEqual(solved.status, "infeasible", solved.message)
                self.assertIsNone(solved.electricity_yuan)

    def test_building_demands_are_not_aggregated_away(self):
        base = tiny_scenario()
        network = replace(base.network, building_station=np.array([0, 0], dtype=np.int64))
        scenario = replace(base, network=network, heat_demand_kw=np.array([[.5, .5], [.5, .5]]))
        solved = solve_dispatch(scenario)
        self.assertEqual(solved.status, "infeasible", solved.message)

    def test_source_capacity_is_station_specific(self):
        base = tiny_scenario()
        repeated = {key: np.repeat(getattr(base.network, key), 2)
                    for key in ("pump_cop", "pump_max_kw", "pump_min_kw", "pump_ramp_kw",
                                "tank_capacity_kwh", "tank_charge_limit_kw", "tank_discharge_limit_kw",
                                "tank_ua_kw_per_k", "tank_heat_capacity_kwh_per_k", "eta_charge",
                                "eta_discharge", "edge_capacity_kw")}
        network = replace(base.network, **repeated,
                          pump_to_tank=np.array([0, 1]), delivery_edges=np.array([[0, 1], [0, 1]]),
                          pump_station=np.array([0, 1]), tank_station=np.array([0, 1]),
                          building_station=np.array([0, 1]), pump_models=("analytic", "analytic"))
        scenario = replace(base, network=network, heat_demand_kw=np.ones((2, 2)),
                           tank_environment_c=np.full((2, 2), 10.),
                           source_limit_kw=np.array([[0., 2.], [0., 2.]]),
                           availability=np.ones((2, 2), dtype=bool), power_upper_kw=np.full(2, 2.),
                           initial_energy_kwh=np.zeros(2), terminal_energy_kwh=np.zeros(2),
                           initial_power_kw=np.zeros(2))
        self.assertEqual(solve_dispatch(scenario).status, "infeasible")

    def test_a_tank_can_supply_multiple_buildings_on_separate_edges(self):
        base = tiny_scenario()
        network = replace(base.network,
                          building_station=np.array([0, 0], dtype=np.int64),
                          delivery_edges=np.array([[0, 0], [0, 1]], dtype=np.int64),
                          edge_capacity_kw=np.array([1., 1.]))
        scenario = replace(base, network=network, heat_demand_kw=np.full((2, 2), .5))
        solved = solve_dispatch(scenario)
        self.assertEqual(solved.status, "optimal", solved.message)
        np.testing.assert_allclose(solved.plan.delivery_kw, .5)
        self.assertAlmostEqual(solved.electricity_yuan, 1.)

    def test_half_hour_interval_converts_power_to_energy_and_money(self):
        solved = solve_dispatch(replace(tiny_scenario(), dt_hours=.5))
        self.assertEqual(solved.status, "optimal", solved.message)
        np.testing.assert_allclose(solved.plan.power_kw[:, 0], [1., 0.])
        np.testing.assert_allclose(solved.plan.energy_kwh[:, 0], [0., .5, 0.])
        self.assertAlmostEqual(solved.electricity_yuan, .5)

    def test_simultaneous_charge_delivery_allows_legal_successor(self):
        base = tiny_scenario()
        scenario = replace(base, initial_energy_kwh=np.array([1.]),
                           terminal_energy_kwh=np.array([1.]), heat_demand_kw=np.full((2, 1), 2.),
                           network=replace(base.network, tank_capacity_kwh=np.array([1.])))
        solved = solve_dispatch(scenario)
        self.assertEqual(solved.status, "optimal", solved.message)
        np.testing.assert_allclose(solved.plan.energy_kwh, 1.)
        self.assertAlmostEqual(solved.electricity_yuan, 4.)

    def test_ground_loss_and_efficiencies_have_correct_signs_and_units(self):
        base = tiny_scenario()
        network = replace(base.network, pump_max_kw=np.array([2.]), pump_ramp_kw=np.array([2.]),
                          tank_charge_limit_kw=np.array([4.]), eta_charge=np.array([.8]),
                          eta_discharge=np.array([.5]), tank_ua_kw_per_k=np.array([.01]))
        scenario = replace(base, network=network, source_limit_kw=np.full((2, 1), 2.),
                           power_upper_kw=np.full(2, 2.), heat_demand_kw=np.zeros((2, 1)),
                           initial_energy_kwh=np.array([1.]), terminal_energy_kwh=np.array([1.]),
                           tank_environment_c=np.full((2, 1), 70.))
        solved = solve_dispatch(scenario, fixed_first_action=Action(np.array([0.]), np.array([0.])))
        self.assertEqual(solved.status, "optimal", solved.message)
        self.assertAlmostEqual(solved.plan.energy_kwh[1, 0], .99)
        self.assertAlmostEqual(solved.plan.power_kw[1, 0], (1. - .99 * .99) / 1.6)

    def test_charging_and_useful_delivery_efficiencies_are_applied_once(self):
        base = tiny_scenario()
        network = replace(base.network, pump_max_kw=np.array([2.]), pump_ramp_kw=np.array([2.]),
                          tank_charge_limit_kw=np.array([4.]), eta_charge=np.array([.8]),
                          eta_discharge=np.array([.5]))
        scenario = replace(base, network=network, source_limit_kw=np.full((2, 1), 2.),
                           power_upper_kw=np.full(2, 2.))
        solved = solve_dispatch(scenario)
        self.assertEqual(solved.status, "optimal", solved.message)
        # Each useful kWh removes two stored kWh; a kWh electricity adds 1.6 stored kWh.
        np.testing.assert_allclose(solved.plan.power_kw[:, 0], [2., .5])
        np.testing.assert_allclose(solved.plan.energy_kwh[:, 0], [0., 1.2, 0.], atol=1e-8)
        self.assertAlmostEqual(solved.electricity_yuan, 3.5)

    def test_unsupported_objectives_fail_before_solver(self):
        for objective in ("energy", "feasibility", "invalid", None, ["electricity"]):
            with self.subTest(objective=objective):
                with self.assertRaisesRegex(ValueError, "objective must be 'electricity'"):
                    build_linear_program(tiny_scenario(), objective=objective)
                with patch("dr_research.optimization.linprog") as mocked:
                    solved = solve_dispatch(tiny_scenario(), objective=objective)
                mocked.assert_not_called()
                self.assertEqual(solved.status, "error")
                self.assertIsNone(solved.plan)
                self.assertIsNone(solved.electricity_yuan)
                self.assertIsNone(solved.lower_bound_yuan)

    def test_bill_optimum_uses_cheap_storage_despite_thermal_losses(self):
        base = tiny_scenario()
        network = replace(base.network, tank_ua_kw_per_k=np.array([.5]))
        scenario = replace(base, network=network, tank_environment_c=np.full((2, 1), 70.))
        bill = solve_dispatch(scenario, objective="electricity")
        self.assertEqual(bill.status, "optimal", bill.message)
        np.testing.assert_allclose(bill.plan.power_kw[:, 0], [1., .25])
        np.testing.assert_allclose(bill.plan.energy_kwh[:, 0], [0., 1., 0.], atol=1e-8)
        self.assertAlmostEqual(bill.electricity_yuan, 1.75)
        # Immediate heating [.5, .5] uses 1 kWh and costs 2 yuan. Preheating
        # uses 1.25 kWh with storage loss but achieves the lower bill above.
        self.assertAlmostEqual(scenario.dt_hours*bill.plan.power_kw.sum(), 1.25)
        self.assertAlmostEqual(bill.lower_bound_yuan, bill.electricity_yuan)

    def test_public_matrices_are_sparse_and_reconstruct_plan(self):
        lp = build_linear_program(tiny_scenario())
        self.assertTrue(isspmatrix_csr(lp.A_eq))
        self.assertTrue(isspmatrix_csr(lp.A_ub))
        self.assertEqual(lp.c.size, 7)
        self.assertEqual(lp.statistics["nnz"], lp.A_eq.nnz + lp.A_ub.nnz)
        self.assertIn("source[0]", lp.constraint_blocks["ub"])
        self.assertIn("building_demand[1]", lp.constraint_blocks["eq"])

    def test_invalid_shapes_and_state_boundaries_fail_before_solver(self):
        scenario = tiny_scenario()
        for state in [State(-1, np.zeros(1), np.zeros(1)),
                      State(3, np.zeros(1), np.zeros(1)),
                      State(0, np.array([np.nan]), np.zeros(1)),
                      State(0, np.array([1.j]), np.zeros(1)),
                      State(0, np.zeros(2), np.zeros(1))]:
            with self.subTest(state=state):
                with patch("dr_research.optimization.linprog") as mocked:
                    solved = solve_dispatch(scenario, state=state)
                    self.assertEqual(solved.status, "error")
                    mocked.assert_not_called()

    def test_solver_limit_is_not_infeasibility(self):
        for raw, label in [(1, "limit"), (2, "infeasible"), (3, "unbounded"), (4, "error")]:
            with self.subTest(raw=raw):
                fake = SimpleNamespace(status=raw, success=False, x=None, nit=0, message="test status")
                with patch("dr_research.optimization.linprog", return_value=fake):
                    solved = solve_dispatch(tiny_scenario())
                self.assertEqual(solved.status, label)
                self.assertIsNone(solved.plan)
                self.assertIsNone(solved.electricity_yuan)
                self.assertIsNone(solved.lower_bound_yuan)

    def test_solver_success_without_independent_validation_is_not_optimal(self):
        report = ValidationReport(False, ("independent reject",), {"test": 1.})
        with patch("dr_research.physics.validate_plan", return_value=report):
            solved = solve_dispatch(tiny_scenario())
        self.assertEqual(solved.status, "error")
        self.assertIsNone(solved.plan)
        self.assertIsNone(solved.lower_bound_yuan)

    def test_limit_with_valid_incumbent_keeps_plan_but_no_optimality_claim(self):
        fake = SimpleNamespace(status=1, success=False, nit=0, message="time limit",
                               x=np.array([1., 0., 1., 1., 0., 1., 0.]))
        with patch("dr_research.optimization.linprog", return_value=fake):
            solved = solve_dispatch(tiny_scenario())
        self.assertEqual(solved.status, "limit")
        self.assertIsNotNone(solved.plan)
        self.assertAlmostEqual(solved.electricity_yuan, 1.)
        self.assertIsNone(solved.lower_bound_yuan)


if __name__ == "__main__":
    unittest.main()

#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.modeling.mhe_constructor import (
    curr_sample_point,
    construct_measurement_variables_constraints,
    construct_disturbed_model_constraints,
    activate_disturbed_constraints_based_on_original_constraints,
    get_cost_from_error_variables,
)
from pyomo.contrib.mpc.data.scalar_data import ScalarData


class TestCurrSamplePoint(unittest.TestCase):

    def test_get_sample_time(self):
        sample_points = [0.0, 2.0, 4.0]

        tp1 = 0
        spt1 = curr_sample_point(tp1, sample_points)
        self.assertEqual(spt1, 0.0)

        tp2 = 2
        spt2 = curr_sample_point(tp2, sample_points)
        self.assertEqual(spt2, 2.0)

        tp3 = 2.5
        spt3 = curr_sample_point(tp3, sample_points)
        self.assertEqual(spt3, 4.0)


class TestConstructMeasurementVariablesConstraints(unittest.TestCase):

    def test_construct_measurement_components(self):
        m = pyo.ConcreteModel()
        m.spts = pyo.Set(initialize=[0.0, 2.0, 4.0])
        m.v1 = pyo.Var(m.spts, initialize={i: 1*i for i in m.spts})
        m.v2 = pyo.Var(m.spts, initialize={i: 2*i for i in m.spts})

        meas = construct_measurement_variables_constraints(
            m.spts, [m.v1, m.v2]
        )
        m.meas_set = meas[0]
        m.meas_var = meas[1]
        m.meas_error_var = meas[2]
        m.meas_con = meas[3]

        self.assertTupleEqual(m.meas_set.ordered_data(), (0,1))
        pred_meas_var_dict = {
            (i,j): None
            for i in m.meas_set for j in m.spts
        }
        self.assertDictEqual(
            m.meas_var.get_values(), pred_meas_var_dict
        )
        pred_meas_error_var_dict = {
            (i,j): 0.0
            for i in m.meas_set for j in m.spts
        }
        self.assertDictEqual(
            m.meas_error_var.get_values(), pred_meas_error_var_dict
        )
        pred_con_expr = {
            (i,j): m.meas_var[i,j] == var[j] + m.meas_error_var[i,j]
            for i, var in enumerate([m.v1,m.v2])
            for j in m.spts
        }
        for i in m.meas_set:
            for j in m.spts:
                self.assertTrue(
                    compare_expressions(
                        pred_con_expr[(i,j)], m.meas_con[(i,j)].expr
                    )
                )


class TestConstructDisturbedModelConstraints(unittest.TestCase):

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[0.0, 1.0, 2.0, 3.0, 4.0])
        m.spts = pyo.Set(initialize=[0.0, 2.0, 4.0])
        m.v1 = pyo.Var(m.time, initialize={i: 1*i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2*i for i in m.time})
        m.v3 = pyo.Var(m.time, initialize={i: 3*i for i in m.time})
        m.c1 = pyo.Constraint(m.time, rule=lambda m,t:m.v1[t]+m.v2[t]<=5)
        m.c2 = pyo.Constraint(m.time, rule=lambda m,t:m.v1[t]+2*m.v2[t]==10)
        m.c3 = pyo.Constraint(m.time, rule=lambda m,t:m.v1[t]+3*m.v2[t]-15==0)
        m.c4 = pyo.Constraint(m.time, rule=lambda m,t:m.v1[t]+4*m.v2[t]==m.v3[t]-5)

        return m

    def test_construct_distrubed_constraints_notEqualityError(self):
        m = self._make_model()

        msg = "Not an equality constraint"
        with self.assertRaisesRegex(RuntimeError, msg):
            dist = construct_disturbed_model_constraints(
                m.time, m.spts, [m.c1, m.c2]
            )
            m.dist_set = dist[0]
            m.dist_var = dist[1]
            m.dist_con = dist[2]

    def test_construct_distrubed_constraints(self):
        m = self._make_model()

        dist = construct_disturbed_model_constraints(
            m.time, m.spts, [m.c2, m.c3, m.c4]
        )
        m.dist_set = dist[0]
        m.dist_var = dist[1]
        m.dist_con = dist[2]

        self.assertTupleEqual(m.dist_set.ordered_data(), (0,1,2,))
        pred_dist_var_dict = {
            (i,j): 0.0
            for i in m.dist_set for j in m.spts
        }
        self.assertDictEqual(
            m.dist_var.get_values(), pred_dist_var_dict
        )
        pred_c2_expr = {
            (0,t): m.v1[t] + 2*m.v2[t] + \
                m.dist_var[0, curr_sample_point(t, m.spts)] == 10
            for t in m.time
        }
        pred_c3_expr = {
            (1,t): m.v1[t] + 3*m.v2[t] - 15 + \
                m.dist_var[1, curr_sample_point(t, m.spts)] == 0
            for t in m.time
        }
        pred_c4_expr = {
            (2,t): m.v1[t] + 4*m.v2[t] - (m.v3[t]-5) + \
                m.dist_var[2, curr_sample_point(t, m.spts)] == 0
            for t in m.time
        }
        pred_con_expr = {**pred_c2_expr, **pred_c3_expr, **pred_c4_expr}
        for i in m.dist_set:
            for j in m.time:
                self.assertTrue(
                    compare_expressions(
                        pred_con_expr[(i,j)], m.dist_con[(i,j)].expr
                    )
                )


class TestActivateDistrubedConstraints(unittest.TestCase):

    def test_activate_disturbed_cons(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[0.0, 1.0, 2.0, 3.0, 4.0])
        m.spts = pyo.Set(initialize=[0.0, 2.0, 4.0])
        m.v1 = pyo.Var(m.time, initialize={i: 1*i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2*i for i in m.time})
        m.c1 = pyo.Constraint(m.time, rule=lambda m,t:m.v1[t]+m.v2[t]<=5)
        m.c2 = pyo.Constraint(m.time, rule=lambda m,t:m.v1[t]+2*m.v2[t]==10)
        m.c2.deactivate()
        m.c3 = pyo.Constraint(m.time, rule=lambda m,t:m.v1[t]+3*m.v2[t]-15==0)
        m.c3[1.0].deactivate()
        m.c3[2.0].deactivate()

        mod_cons = [m.c2, m.c3]
        dist = construct_disturbed_model_constraints(
            m.time, m.spts, mod_cons
        )
        m.dist_set = dist[0]
        m.dist_var = dist[1]
        m.dist_con = dist[2]

        activate_disturbed_constraints_based_on_original_constraints(
            m.time, m.spts, m.dist_var, mod_cons, m.dist_con
        )

        for t in m.time:
            self.assertFalse(m.dist_con[0,t].active)
        self.assertFalse(m.dist_con[1, 1.0].active)
        self.assertFalse(m.dist_con[1, 2.0].active)

        for sp in m.spts:
            self.assertTrue(m.dist_var[0, sp].fixed)
        self.assertTrue(m.dist_var[1, 2.0].fixed)


class TestGetCostFromErrorVariables(unittest.TestCase):

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=[0.0, 1.0, 2.0])
        m.v1 = pyo.Var(m.time, initialize={i: 1*i for i in m.time})
        m.v2 = pyo.Var(m.time, initialize={i: 2*i for i in m.time})

        return m

    def test_get_cost_from_error_no_weights(self):
        m = self._make_model()
        m.error_cost = get_cost_from_error_variables(
            [m.v1, m.v2], m.time,
        )

        var_sets = {
            i: ComponentSet(identify_variables(m.error_cost[i]))
            for i in m.time
        }
        for i in m.time:
            self.assertIn(m.v1[i], var_sets[i])
            self.assertIn(m.v2[i], var_sets[i])
            pred_value = (1*i)**2 + (2*i)**2
            self.assertEqual(pred_value, pyo.value(m.error_cost[i]))
            pred_expr = (m.v1[i])**2 + (m.v2[i])**2
            self.assertTrue(compare_expressions(
                pred_expr, m.error_cost[i].expr
            ))

    def test_get_cost_from_error_with_weights(self):
        m = self._make_model()
        weight_data = ScalarData({m.v1[:]: 0.1, m.v2[:]: 0.5})
        m.error_cost = get_cost_from_error_variables(
            [m.v1, m.v2], m.time, weight_data,
        )

        var_sets = {
            i: ComponentSet(identify_variables(m.error_cost[i]))
            for i in m.time
        }
        for i in m.time:
            self.assertIn(m.v1[i], var_sets[i])
            self.assertIn(m.v2[i], var_sets[i])
            pred_value = 0.1*(1*i)**2 + 0.5*(2*i)**2
            self.assertEqual(pred_value, pyo.value(m.error_cost[i]))
            pred_expr = 0.1*(m.v1[i])**2 + 0.5*(m.v2[i])**2
            self.assertTrue(compare_expressions(
                pred_expr, m.error_cost[i].expr
            ))


if __name__ == "__main__":
    unittest.main()
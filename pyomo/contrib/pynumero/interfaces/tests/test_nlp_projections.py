#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
import os

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run NLP tests")

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import RenamedNLP, ProjectedNLP

def create_pyomo_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(3), bounds=(-10,10), initialize={0:1.0, 1:2.0, 2:4.0})

    m.obj = pyo.Objective(expr=m.x[0]**2 + m.x[0]*m.x[1] + m.x[0]*m.x[2] + m.x[2]**2)

    m.con1 = pyo.Constraint(expr=m.x[0]*m.x[1] + m.x[0]*m.x[2] == 4)
    m.con2 = pyo.Constraint(expr=m.x[0] + m.x[2] == 4)

    return m

class TestRenamedNLP(unittest.TestCase):
    def test_rename(self):
        m = create_pyomo_model()
        nlp = PyomoNLP(m)
        expected_names = ['x[0]', 'x[1]', 'x[2]']
        self.assertEqual(nlp.primals_names(), expected_names)
        renamed_nlp = RenamedNLP(nlp, {'x[0]': 'y[0]', 'x[1]':'y[1]', 'x[2]':'y[2]'})
        expected_names = ['y[0]', 'y[1]', 'y[2]']
        
class TestProjectedNLP(unittest.TestCase):
    def test_projected(self):
        m = create_pyomo_model()
        nlp = PyomoNLP(m)
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[1]', 'x[2]'])
        expected_names = ['x[0]', 'x[1]', 'x[2]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        self.assertTrue(np.array_equal(projected_nlp.get_primals(),
                                       np.asarray([1.0, 2.0, 4.0])))
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(),
                                       np.asarray([8.0, 1.0, 9.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 1.0],[1.0, 0.0, 1.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0*J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 1.0],[1.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0*H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # now test a reordering
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]', 'x[1]'])
        expected_names = ['x[0]', 'x[2]', 'x[1]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        self.assertTrue(np.array_equal(projected_nlp.get_primals(), np.asarray([1.0, 4.0, 2.0])))
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(),
                                       np.asarray([8.0, 9.0, 1.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 1.0],[1.0, 1.0, 0.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0*J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 1.0],[1.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0*H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # now test an expansion
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]', 'y', 'x[1]'])
        expected_names = ['x[0]', 'x[2]', 'y', 'x[1]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        np.testing.assert_equal(projected_nlp.get_primals(),np.asarray([1.0, 4.0, np.nan, 2.0]))
        
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(),
                                       np.asarray([8.0, 9.0, 0.0, 1.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 5)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 6)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 5)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0, 0.0, 1.0],[1.0, 1.0, 0.0, 0.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0*J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 6)
        expectedH = np.asarray([[2.0, 1.0, 0.0, 1.0],[1.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0*H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # now test an expansion
        projected_nlp = ProjectedNLP(nlp, ['x[0]', 'x[2]'])
        expected_names = ['x[0]', 'x[2]']
        self.assertEqual(projected_nlp.primals_names(), expected_names)
        np.testing.assert_equal(projected_nlp.get_primals(),np.asarray([1.0, 4.0]))
        
        self.assertTrue(np.array_equal(projected_nlp.evaluate_grad_objective(),
                                       np.asarray([8.0, 9.0])))
        self.assertEqual(projected_nlp.nnz_jacobian(), 4)
        self.assertEqual(projected_nlp.nnz_hessian_lag(), 4)

        J = projected_nlp.evaluate_jacobian()
        self.assertEqual(len(J.data), 4)
        denseJ = J.todense()
        expected_jac = np.asarray([[6.0, 1.0],[1.0, 1.0]])
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        # test the use of "out"
        J = 0.0*J
        projected_nlp.evaluate_jacobian(out=J)
        denseJ = J.todense()
        self.assertTrue(np.array_equal(denseJ, expected_jac))

        H = projected_nlp.evaluate_hessian_lag()
        self.assertEqual(len(H.data), 4)
        expectedH = np.asarray([[2.0, 1.0],[1.0, 2.0]])
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))

        # test the use of "out"
        H = 0.0*H
        projected_nlp.evaluate_hessian_lag(out=H)
        denseH = H.todense()
        self.assertTrue(np.array_equal(denseH, expectedH))


class TestProjectConstraints(unittest.TestCase):

    def _make_simple_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([0, 1, 2, 3], initialize={0: 1.1, 1: 1.2, 2: 1.3, 3: 1.4})
        m.con1 = pyo.Constraint(expr=m.x[1]**2 == 5.0)
        m.con2 = pyo.Constraint(expr=m.x[1] + 1.5*m.x[0]**2 + 3*m.x[2] == 1.0)
        m.con3 = pyo.Constraint(
            expr=1.1*m.x[1] + 2*m.x[0] - 1.2*m.x[2] + 1.3*m.x[3]**3 == 2.0
        )
        m.obj = pyo.Objective(expr=sum(var**2 for var in m.x.values()))
        return m

    def _test_solve_ipopt(self):
        m = self._make_simple_model()
        ipopt = pyo.SolverFactory("ipopt")
        ipopt.solve(m, tee=True)

    def test_project_nlp_evaluate_1constraint(self):
        m = self._make_simple_model()
        original_nlp = PyomoNLP(m)
        primals_ordering = [
            var.name for var in original_nlp.get_pyomo_variables()
        ]
        constraints_ordering = original_nlp.get_constraint_indices([m.con2])
        proj_nlp = ProjectedNLP(
            original_nlp,
            primals_ordering,
            constraints_ordering=constraints_ordering,
        )
        n_primals_orig = original_nlp.n_primals()
        self.assertEqual(proj_nlp.n_primals(), n_primals_orig)
        self.assertEqual(proj_nlp.n_constraints(), len(constraints_ordering))

        orig_con = original_nlp.evaluate_constraints()
        proj_con = proj_nlp.evaluate_constraints()
        self.assertEqual(len(proj_con), len(constraints_ordering))
        # From con2 with initial variable values:
        self.assertEqual(proj_con[0], 1.2 + 1.5*1.1**2 + 3*1.3 - 1.0)
        for i_proj, i_orig in enumerate(constraints_ordering):
            self.assertEqual(proj_con[i_proj], orig_con[i_orig])

    def test_project_nlp_evaluate_2constraints(self):
        m = self._make_simple_model()
        original_nlp = PyomoNLP(m)
        primals_ordering = [
            var.name for var in original_nlp.get_pyomo_variables()
        ]
        constraints_ordering = original_nlp.get_constraint_indices(
            [m.con3, m.con2]
        )
        proj_nlp = ProjectedNLP(
            original_nlp,
            primals_ordering,
            constraints_ordering=constraints_ordering,
        )
        n_primals_orig = original_nlp.n_primals()
        self.assertEqual(proj_nlp.n_primals(), n_primals_orig)
        self.assertEqual(proj_nlp.n_constraints(), len(constraints_ordering))

        # With returned value
        orig_con = original_nlp.evaluate_constraints()
        proj_con = proj_nlp.evaluate_constraints()
        self.assertEqual(len(proj_con), len(constraints_ordering))
        for i_proj, i_orig in enumerate(constraints_ordering):
            self.assertEqual(proj_con[i_proj], orig_con[i_orig])

        # Storing the output value in an input array
        out = np.zeros(len(constraints_ordering))
        proj_nlp.evaluate_constraints(out)
        for i_proj, i_orig in enumerate(constraints_ordering):
            self.assertEqual(out[i_proj], orig_con[i_orig])

    def test_project_nlp_jacobian_1constraint(self):
        m = self._make_simple_model()
        original_nlp = PyomoNLP(m)
        primals_ordering = [
            # We don't re-order or project primals here, but this is a
            # required argument to ProjectedNLP
            var.name for var in original_nlp.get_pyomo_variables()
        ]
        constraints_ordering = original_nlp.get_constraint_indices([m.con2])
        # Create projected NLP from original NLP.
        proj_nlp = ProjectedNLP(
            original_nlp,
            primals_ordering,
            constraints_ordering=constraints_ordering,
        )
        # Make sure we have the right number of variables and constraints
        n_primals_orig = original_nlp.n_primals()
        self.assertEqual(proj_nlp.n_primals(), n_primals_orig)
        n_con_proj = len(constraints_ordering)
        self.assertEqual(proj_nlp.n_constraints(), n_con_proj)

        # Get original and projected Jacobians
        orig_jac = original_nlp.evaluate_jacobian()
        proj_jac = proj_nlp.evaluate_jacobian()
        # Make sure projected Jacobian has right shape
        self.assertEqual(proj_jac.shape, (n_con_proj, n_primals_orig))

        constraint_coord_set = set(constraints_ordering)
        nz_to_retain = [(i, j) for i, j in zip(orig_jac.row, orig_jac.col)
                if i in constraint_coord_set]
        nz_to_retain_set = set(nz_to_retain)
        nz_dict = {
            (i, j): d for i, j, d in 
            zip(orig_jac.row, orig_jac.col, orig_jac.data)
            if (i, j) in nz_to_retain_set
        }
        # Make sure projected Jacobian has expected nonzero structure
        # and values
        self.assertEqual(len(proj_jac.data), len(nz_to_retain))
        for i, j, d in zip(proj_jac.row, proj_jac.col, proj_jac.data):
            # Map projected index to old index
            i = constraints_ordering[i]
            pred_val = nz_dict[i, j]
            self.assertEqual(pred_val, d)

        # Sanity check
        pred_data_set = {1.5*2*1.1, 1.0, 3.0}
        self.assertEqual(set(proj_jac.data), pred_data_set)

    def test_project_nlp_jacobian_2constraints(self):
        m = self._make_simple_model()
        original_nlp = PyomoNLP(m)
        primals_ordering = [
            # We don't re-order or project primals here, but this is a
            # required argument to ProjectedNLP
            var.name for var in original_nlp.get_pyomo_variables()
        ]
        constraints_ordering = original_nlp.get_constraint_indices(
            [m.con3, m.con2]
        )
        # Create projected NLP from original NLP.
        proj_nlp = ProjectedNLP(
            original_nlp,
            primals_ordering,
            constraints_ordering=constraints_ordering,
        )
        # Make sure we have the right number of variables and constraints
        n_primals_orig = original_nlp.n_primals()
        self.assertEqual(proj_nlp.n_primals(), n_primals_orig)
        n_con_proj = len(constraints_ordering)
        self.assertEqual(proj_nlp.n_constraints(), n_con_proj)

        # Get original and projected Jacobians
        orig_jac = original_nlp.evaluate_jacobian()
        proj_jac = proj_nlp.evaluate_jacobian()
        # Make sure projected Jacobian has right shape
        self.assertEqual(proj_jac.shape, (n_con_proj, n_primals_orig))

        constraint_coord_set = set(constraints_ordering)
        nz_to_retain = [(i, j) for i, j in zip(orig_jac.row, orig_jac.col)
                if i in constraint_coord_set]
        nz_to_retain_set = set(nz_to_retain)
        nz_dict = {
            (i, j): d for i, j, d in 
            zip(orig_jac.row, orig_jac.col, orig_jac.data)
            if (i, j) in nz_to_retain_set
        }
        # Make sure projected Jacobian has expected nonzero structure
        # and values
        self.assertEqual(len(proj_jac.data), len(nz_to_retain))
        for i, j, d in zip(proj_jac.row, proj_jac.col, proj_jac.data):
            # Map projected index to old index
            i = constraints_ordering[i]
            pred_val = nz_dict[i, j]
            self.assertEqual(pred_val, d)

        # Sanity check
        # This test no longer works (once I added nonlinearities) due to
        # roundoff error.
        #self.assertEqual(len(proj_jac.data), 7)
        #pred_data_set = {1.0, 1.5*2*1.1, 3.0, 1.1, -1.2, 1.3*1.4**2*3, 2.0}
        #self.assertEqual(set(proj_jac.data), pred_data_set)

    def test_project_nlp_duals_1constraint(self):
        m = self._make_simple_model()
        original_nlp = PyomoNLP(m)
        primals_ordering = [
            # We don't re-order or project primals here, but this is a
            # required argument to ProjectedNLP
            var.name for var in original_nlp.get_pyomo_variables()
        ]
        # Maps new index to old index
        constraints_ordering = original_nlp.get_constraint_indices([m.con2])
        # Create projected NLP from original NLP.
        proj_nlp = ProjectedNLP(
            original_nlp,
            primals_ordering,
            constraints_ordering=constraints_ordering,
        )
        # Make sure we have the right number of variables and constraints
        n_primals_orig = original_nlp.n_primals()
        self.assertEqual(proj_nlp.n_primals(), n_primals_orig)
        n_con_proj = len(constraints_ordering)
        self.assertEqual(proj_nlp.n_constraints(), n_con_proj)

        # Make sure we get duals from the projected NLP properly
        duals = [1.0 + 0.1*i for i in range(original_nlp.n_constraints())]
        original_nlp.set_duals(duals)
        proj_duals = proj_nlp.get_duals()
        for i, dual in enumerate(proj_duals):
            self.assertEqual(dual, duals[constraints_ordering[i]])

        # Sanity check
        # This relies on con2 being the second constraint in
        # get_constraint_indices
        self.assertEqual(proj_duals[0], 1.1)

        # Make sure, when we set duals, we can get them properly in either
        # NLP.
        old_duals = duals
        new_duals = [2.0 + 0.1*i for i in range(proj_nlp.n_constraints())]
        proj_nlp.set_duals(new_duals)
        duals_in_orig = original_nlp.get_duals()
        proj_duals = proj_nlp.get_duals()
        for i, dual in enumerate(proj_duals):
            self.assertEqual(dual, duals_in_orig[constraints_ordering[i]])
            self.assertEqual(dual, new_duals[i])

        # Sanity check
        self.assertEqual(proj_duals[0], 2.0)

        # Make sure coordinates that we have not used in the projection
        # are unaltered.
        retained_coord_set = set(constraints_ordering)
        for i in range(original_nlp.n_constraints()):
            if i not in retained_coord_set:
                self.assertEqual(duals_in_orig[i], old_duals[i])

    def test_project_nlp_duals_2constraints(self):
        m = self._make_simple_model()
        original_nlp = PyomoNLP(m)
        primals_ordering = [
            # We don't re-order or project primals here, but this is a
            # required argument to ProjectedNLP
            var.name for var in original_nlp.get_pyomo_variables()
        ]
        # Maps new index to old index
        constraints_ordering = original_nlp.get_constraint_indices(
            [m.con3, m.con2]
        )
        # Create projected NLP from original NLP.
        proj_nlp = ProjectedNLP(
            original_nlp,
            primals_ordering,
            constraints_ordering=constraints_ordering,
        )
        # Make sure we have the right number of variables and constraints
        n_primals_orig = original_nlp.n_primals()
        self.assertEqual(proj_nlp.n_primals(), n_primals_orig)
        n_con_proj = len(constraints_ordering)
        self.assertEqual(proj_nlp.n_constraints(), n_con_proj)

        # Make sure we get duals from the projected NLP properly
        duals = [1.0 + 0.1*i for i in range(original_nlp.n_constraints())]
        original_nlp.set_duals(duals)
        proj_duals = proj_nlp.get_duals()
        for i, dual in enumerate(proj_duals):
            self.assertEqual(dual, duals[constraints_ordering[i]])

        # Make sure, when we set duals, we can get them properly in either
        # NLP.
        old_duals = duals
        new_duals = [2.0 + 0.1*i for i in range(proj_nlp.n_constraints())]
        proj_nlp.set_duals(new_duals)
        duals_in_orig = original_nlp.get_duals()
        proj_duals = proj_nlp.get_duals()
        for i, dual in enumerate(proj_duals):
            self.assertEqual(dual, duals_in_orig[constraints_ordering[i]])
            self.assertEqual(dual, new_duals[i])

        # Make sure coordinates that we have not used in the projection
        # are unaltered.
        retained_coord_set = set(constraints_ordering)
        for i in range(original_nlp.n_constraints()):
            if i not in retained_coord_set:
                self.assertEqual(duals_in_orig[i], old_duals[i])

    def test_project_nlp_hessian_1constraint(self):
        m = self._make_simple_model()
        original_nlp = PyomoNLP(m)
        primals_ordering = [
            # We don't re-order or project primals here, but this is a
            # required argument to ProjectedNLP
            var.name for var in original_nlp.get_pyomo_variables()
        ]
        # Maps new index to old index
        constraints_ordering = original_nlp.get_constraint_indices([m.con2])
        # Create projected NLP from original NLP.
        proj_nlp = ProjectedNLP(
            original_nlp,
            primals_ordering,
            constraints_ordering=constraints_ordering,
        )
        # Make sure we have the right number of variables and constraints
        n_primals_orig = original_nlp.n_primals()
        self.assertEqual(proj_nlp.n_primals(), n_primals_orig)
        n_con_proj = len(constraints_ordering)
        self.assertEqual(proj_nlp.n_constraints(), n_con_proj)

        orig_duals = [-1.0 for _ in range(original_nlp.n_constraints())]
        original_nlp.set_duals(orig_duals)
        duals = [1.0 + 0.1*(i+1) for i in range(n_con_proj)]
        proj_nlp.set_duals(duals)
        proj_hess = proj_nlp.evaluate_hessian_lag()
        x0_coord = original_nlp.get_primal_indices([m.x[0]])[0]

        # Hard-code Hessian we expect
        pred_hess = np.zeros((n_primals_orig, n_primals_orig))
        pred_hess[x0_coord, x0_coord] = duals[0]*1.5*2
        np.testing.assert_allclose(pred_hess, proj_hess.toarray())

        # Hessian has entries due to all constraints and objective
        # due to our hacky way of getting Hessian-of-subset-of-constraints
        # using ASL.
        pred_hess_dict = {}
        pred_hess_dict[x0_coord, x0_coord] = duals[0]*1.5*2
        for i in range(n_primals_orig):
            if i != x0_coord:
                pred_hess_dict[i, i] = 0.0
        self.assertEqual(len(pred_hess_dict), len(proj_hess.data))
        for i, j, d in zip(proj_hess.row, proj_hess.col, proj_hess.data):
            self.assertAlmostEqual(pred_hess_dict[i, j], d)

        # Make sure duals in original NLP are what we expect
        duals_in_orig = original_nlp.get_duals()
        pred_orig_duals = np.array(orig_duals)
        for i, dual in enumerate(duals):
            # These are the duals we sent to the projected NLP
            pred_orig_duals[constraints_ordering[i]] = dual
        np.testing.assert_array_equal(duals_in_orig, pred_orig_duals)

    def test_project_nlp_hessian_2constraints(self):
        m = self._make_simple_model()
        original_nlp = PyomoNLP(m)
        primals_ordering = [
            # We don't re-order or project primals here, but this is a
            # required argument to ProjectedNLP
            var.name for var in original_nlp.get_pyomo_variables()
        ]
        # Maps new index to old index
        constraints_ordering = original_nlp.get_constraint_indices(
            [m.con3, m.con2]
        )
        # Create projected NLP from original NLP.
        proj_nlp = ProjectedNLP(
            original_nlp,
            primals_ordering,
            constraints_ordering=constraints_ordering,
        )
        # Make sure we have the right number of variables and constraints
        n_primals_orig = original_nlp.n_primals()
        self.assertEqual(proj_nlp.n_primals(), n_primals_orig)
        n_con_proj = len(constraints_ordering)
        self.assertEqual(proj_nlp.n_constraints(), n_con_proj)

        orig_duals = [-1.0 for _ in range(original_nlp.n_constraints())]
        original_nlp.set_duals(orig_duals)
        duals = [1.0 + 0.1*(i+1) for i in range(n_con_proj)]
        proj_nlp.set_duals(duals)
        proj_hess = proj_nlp.evaluate_hessian_lag()
        x_coord = original_nlp.get_primal_indices(
            [m.x[0], m.x[1], m.x[2], m.x[3]]
        )

        # Hard-code Hessian we expect
        pred_hess = np.zeros((n_primals_orig, n_primals_orig))
        pred_hess[x_coord[0], x_coord[0]] = duals[1]*1.5*2
        pred_hess[x_coord[3], x_coord[3]] = duals[0]*1.3*3*2*1.4
        np.testing.assert_allclose(pred_hess, proj_hess.toarray())

        # Hessian has entries due to all constraints and objective
        # due to our hacky way of getting Hessian-of-subset-of-constraints
        # using ASL.
        pred_hess_dict = {}
        pred_hess_dict[x_coord[0], x_coord[0]] = duals[1]*1.5*2
        pred_hess_dict[x_coord[3], x_coord[3]] = duals[0]*1.3*3*2*1.4
        for i in range(n_primals_orig):
            if i != x_coord[0] and i != x_coord[3]:
                pred_hess_dict[i, i] = 0.0
        self.assertEqual(len(pred_hess_dict), len(proj_hess.data))
        for i, j, d in zip(proj_hess.row, proj_hess.col, proj_hess.data):
            self.assertAlmostEqual(pred_hess_dict[i, j], d)

        # Make sure duals in original NLP are what we expect
        duals_in_orig = original_nlp.get_duals()
        pred_orig_duals = np.array(orig_duals)
        for i, dual in enumerate(duals):
            # These are the duals we sent to the projected NLP
            pred_orig_duals[constraints_ordering[i]] = dual
        np.testing.assert_array_equal(duals_in_orig, pred_orig_duals)


if __name__ == '__main__':
    #TestRenamedNLP().test_rename()
    #TestProjectedNLP().test_projected()
    TestProjectConstraints()._test_solve_ipopt()
    TestProjectConstraints().test_project_nlp_evaluate_1constraint()
    TestProjectConstraints().test_project_nlp_evaluate_2constraints()
    TestProjectConstraints().test_project_nlp_jacobian_1constraint()
    TestProjectConstraints().test_project_nlp_jacobian_2constraints()
    TestProjectConstraints().test_project_nlp_duals_1constraint()
    TestProjectConstraints().test_project_nlp_duals_2constraints()
    TestProjectConstraints().test_project_nlp_hessian_1constraint()
    TestProjectConstraints().test_project_nlp_hessian_2constraints()

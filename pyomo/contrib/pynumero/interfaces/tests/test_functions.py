#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy, scipy_available
)

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.common.dependencies.scipy import sparse as sps

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run cyipopt tests")

from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    cyipopt_available,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxModel,
    ExternalGreyBoxBlock,
    ScalarExternalGreyBoxBlock,
    IndexedExternalGreyBoxBlock,
)
from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
    PressureDropTwoOutputsWithHessian,
)
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import (
    PyomoNLP,
)
from pyomo.contrib.pynumero.interfaces.functions import (
    VectorValuedExternalFunction,
    FunctionComposition,
    FunctionFromNLP,
    FunctionStack,
    FunctionCombination,
    IdentityFunction,
    NLPFromFunction,
)
from pyomo.contrib.pynumero.interfaces.abstract_nlps import (
    FixedVarNLP,
)
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    CyIpoptNLP,
    CyIpoptSolver,
)

if not pyo.SolverFactory("ipopt").available():
    raise unittest.SkipTest(
        "Need IPOPT to run ExternalPyomoModel tests"
        )


class SimpleFunction1(VectorValuedExternalFunction):
    """
    Maps x to (5*x - 1, x**2 + 50)
    """

    def __init__(self):
        self.inputs = np.zeros(1)
        self.outputs = np.zeros(2)
        self.jacobian_outputs = np.zeros((2, 1))
        self.hessian_outputs = np.zeros((2, 1, 1))

    def n_inputs(self):
        return len(self.inputs)

    def n_outputs(self):
        return len(self.outputs)

    def set_input_values(self, input_values):
        self.inputs = input_values
        x = self.inputs[0]
        self.outputs[0] = 5.0*x - 1.0
        self.outputs[1] = x**2 + 50.0
        self.jacobian_outputs[0, 0] = 5.0
        self.jacobian_outputs[1, 0] = 2.0*x
        h000 = 0.0
        h100 = 2.0
        self.hessian_outputs[0, 0, 0] = h000
        self.hessian_outputs[1, 0, 0] = h100

    def evaluate_outputs(self):
        return self.outputs

    def evaluate_jacobian_outputs(self):
        return sps.coo_matrix(self.jacobian_outputs)

    def evaluate_hessian_outputs(self):
        return [sps.coo_matrix(self.hessian_outputs[i, :, :])
            for i in range(self.n_outputs())]


class SimpleFunction2(VectorValuedExternalFunction):
    """
    My vision for a vector-valued external function is an object
    with no notion of equality constraints, and no notion of names.
    Ideally, it performs all calculations in compiled code.
    """

    def __init__(self):
        self.inputs = np.zeros(2)
        self.outputs = np.zeros(1)
        self.jacobian_outputs = np.zeros((1, 2))
        self.hessian_outputs = np.zeros((1, 2, 2))

    def n_inputs(self):
        return len(self.inputs)

    def set_input_values(self, input_values):
        self.inputs = input_values
        self.outputs[0] = np.sqrt(self.inputs[0]**2 + self.inputs[1]**2)
        self.jacobian_outputs[0, 0] = self.inputs[0]/self.outputs[0]
        self.jacobian_outputs[0, 1] = self.inputs[1]/self.outputs[0]
        h000 = (self.outputs[0]**2 - self.inputs[0]**2)/self.outputs[0]**3
        h001 = - self.inputs[0]*self.inputs[1]/self.outputs[0]**3
        h011 = (self.outputs[0]**2 - self.inputs[1]**2)/self.outputs[0]**3
        self.hessian_outputs[0, 0, 0] = h000
        self.hessian_outputs[0, 1, 0] = h001
        self.hessian_outputs[0, 0, 1] = h001
        self.hessian_outputs[0, 1, 1] = h011

    def n_outputs(self):
        return len(self.outputs)

    def evaluate_outputs(self):
        return self.outputs

    def evaluate_jacobian_outputs(self):
        return sps.coo_matrix(self.jacobian_outputs)

    def evaluate_hessian_outputs(self):
        # TODO: What data format should I use for sparse tensors.
        # Want:
        # - fast access along rank 1 (to get an individual output's hessian matrix)
        # - Fast multiplication and transpose for a slice across rank 1
        # - Identification of a slice across ranks 2 and 3. This is what
        #   an array of coo matrices doesn't handle well...
        # Current choice is a list of coo matrices.
        return [sps.coo_matrix(self.hessian_outputs[i, :, :])
            for i in range(self.n_outputs())]


"""
The following four functions define models used to test
the embedding of a scalar-valued function in an NLP.
"""


def make_model1_xu():
    """
    Model in the space of x and u.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.5)
    m.u = pyo.Var(initialize=2.5)
    m.eq_con = pyo.Constraint(expr=m.x*m.u - 1.0 == 0)
    m.obj = pyo.Objective(expr=m.x**2 + 2*m.u**2)
    return m


def make_model1_yzu():
    """
    Suppose x is a function of y and z.
    x = sqrt(y**2 + z**2)
    This model is in the space of u, y, and z
    """
    m = pyo.ConcreteModel()
    m.y = pyo.Var(initialize=1.5)
    m.z = pyo.Var(initialize=1.5)
    m.u = pyo.Var(initialize=2.5)
    m.eq_con = pyo.Constraint(expr=pyo.sqrt(m.y**2 + m.z**2)*m.u - 1.0 == 0)
    m.obj = pyo.Objective(expr=m.y**2 + m.z**2 + 2*m.u**2)
    return m


def make_model2_xyzu():
    """
    Suppose y and z are already used within our model.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(initialize=1.5)
    m.y = pyo.Var(initialize=1.5)
    m.z = pyo.Var(initialize=1.5)
    m.u = pyo.Var(initialize=2.5)
    m.eq_con1 = pyo.Constraint(expr=m.x*m.u - 1.0 == 0)
    m.eq_con2 = pyo.Constraint(expr=m.y*m.z - 2.0 == 0)
    m.obj = pyo.Objective(expr=m.x**2 + 2*m.u**2)
    return m


def make_model2_yzu():
    m = pyo.ConcreteModel()
    m.y = pyo.Var(initialize=1.5)
    m.z = pyo.Var(initialize=1.5)
    m.u = pyo.Var(initialize=2.5)
    m.eq_con1 = pyo.Constraint(expr=pyo.sqrt(m.y**2 + m.z**2)*m.u - 1.0 == 0)
    m.eq_con2 = pyo.Constraint(expr=m.y*m.z - 2.0 == 0)
    m.obj = pyo.Objective(expr=m.y**2 + m.z**2 + 2*m.u**2)
    return m


class TestSimpleFunctionComposition(unittest.TestCase):

    def test_outputs(self):
        f = SimpleFunction1()
        g = SimpleFunction2()
        fog = FunctionComposition(f, g)

        inputs = np.array([1.0, 2.0])
        fog.set_input_values(inputs)
        outputs = fog.evaluate_outputs()

        np.testing.assert_allclose(
                outputs,
                [np.sqrt(5.0)*5.0 - 1.0, 55.0],
                )

    def test_jacobian(self):
        f = SimpleFunction1()
        g = SimpleFunction2()
        fog = FunctionComposition(f, g)

        y, z = 1.0, 2.0
        inputs = np.array([y, z])
        fog.set_input_values(inputs)
        jacobian = fog.evaluate_jacobian_outputs()
        
        jac_pred = [
                [5*y/np.sqrt(y**2 + z**2), 5*z/np.sqrt(y**2 + z**2)],
                [2*y, 2*z],
                ]
        np.testing.assert_allclose(jacobian.toarray(), jac_pred)

    def test_hessian(self):
        f = SimpleFunction1()
        g = SimpleFunction2()
        fog = FunctionComposition(f, g)

        y, z = 1.0, 2.0
        inputs = np.array([y, z])
        fog.set_input_values(inputs)
        hessian = fog.evaluate_hessian_outputs()

        denom = np.sqrt(y**2+z**2)**3
        hess_pred = [
                [[5*z**2/denom, -5*y*z/denom],
                [-5*y*z/denom, 5*y**2/denom]],
                [[2.0, 0.0],
                [0.0, 2.0]],
                ]
        for pred, act in zip(hess_pred, hessian):
            np.testing.assert_allclose(pred, act.toarray())


class TestFunctionFromNLP(unittest.TestCase):

    def test_outputs(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        fcn = FunctionFromNLP(nlp)

        self.assertEqual(fcn.n_inputs(), 2)
        self.assertEqual(fcn.n_outputs(), 2)

        x, u = 5.0, 4.0
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        input_values = np.zeros(fcn.n_inputs())
        input_values[x_idx] = x
        input_values[u_idx] = u
        fcn.set_input_values(input_values)

        outputs = fcn.evaluate_outputs()

        # Know that the objective comes before the (single) constraint,
        # so don't have to get any indices here.
        pred_outputs = [57., 19.]
        np.testing.assert_allclose(outputs, pred_outputs)

    def test_jacobian(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        fcn = FunctionFromNLP(nlp)

        x, u = 5.0, 4.0
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        input_values = np.zeros(2)
        input_values[x_idx] = x
        input_values[u_idx] = u
        fcn.set_input_values(input_values)

        jacobian = fcn.evaluate_jacobian_outputs()
        pred_jac = np.zeros((2,2))
        pred_jac[0, x_idx] = 2*x
        pred_jac[0, u_idx] = 4*u
        pred_jac[1, x_idx] = u
        pred_jac[1, u_idx] = x
        np.testing.assert_allclose(jacobian.toarray(), pred_jac)

    def test_hessian(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        fcn = FunctionFromNLP(nlp)

        x, u = 5.0, 4.0
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        input_values = np.zeros(2)
        input_values[x_idx] = x
        input_values[u_idx] = u
        fcn.set_input_values(input_values)

        hessian = fcn.evaluate_hessian_outputs()
        pred_hess = [
            [[2., 0.], [0., 4.]],
            [[0., 1.], [1., 0.]],
            ]
        for pred, act in zip(pred_hess, hessian):
            np.testing.assert_allclose(pred, act.toarray())


class TestComposeFunctionFromNLP(unittest.TestCase):

    def test_outputs(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        self.assertEqual(to_embed.n_inputs(), 3)
        self.assertEqual(to_embed.n_outputs(), 2)

        y, z, u = 2.0, 3.0, 4.0
        # Need a way to map my inputs to their coordinates
        # I.e. need the offset
        y_idx_f = to_embed.get_input_offset(x_idx)
        z_idx_f = y_idx_f + 1
        u_idx_f = to_embed.get_input_offset(u_idx)
        inputs = np.zeros(3)
        inputs[y_idx_f] = y
        inputs[z_idx_f] = z
        inputs[u_idx_f] = u

        to_embed.set_input_values(inputs)
        intermed_outputs = to_embed.evaluate_outputs()
        # Now need a way to get output offsets.
        # Easy here because my outputs are one-dimensional
        pred_outputs = [None, None]
        # These indices could be different than x_idx and u_idx.
        # x_idx and u_idx are just indices into the user-provided
        # list of functions.
        # These indices are actual inputs into the nlp.
        x_idx_nlp = to_embed.get_output_offset(x_idx)
        u_idx_nlp = to_embed.get_output_offset(u_idx)
        pred_outputs[x_idx] = np.sqrt(y**2 + z**2)
        pred_outputs[u_idx] = u
        np.testing.assert_allclose(intermed_outputs, pred_outputs)

        fcn_comp = FunctionComposition(nlp_fcn, to_embed)
        fcn_comp.set_input_values(inputs)
        nlp_outputs = fcn_comp.evaluate_outputs()
        # I know these coordinates because my NLP only has two
        # "row coordinates." If it had multiple constraint functions,
        # I would use get_constraint_indices.
        pred_outputs = [
            y**2 + z**2 + 2*u**2,
            u*np.sqrt(y**2 + z**2) - 1.0,
        ]
        np.testing.assert_allclose(nlp_outputs, pred_outputs)

    def test_jacobian(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        # Is this really the best way order the functions I need to
        # send to the NLP?
        # ... Ideally I re-order the NLP the match some partition
        # in my model...
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        y, z, u = 2.0, 3.0, 4.0
        inputs = np.array([y, z, u])
        to_embed.set_input_values(inputs)
        intermed_jac = to_embed.evaluate_jacobian_outputs()
        # Now need a way to get output offsets.
        # Easy here because my outputs are one-dimensional
        pred_jac = np.zeros((2, 3))
        y_idx_f = to_embed.get_input_offset(0)
        z_idx_f = y_idx_f + 1
        u_idx_f = to_embed.get_input_offset(1)
        denom = np.sqrt(y**2 + z**2)
        pred_jac[x_idx][y_idx_f] = y/denom
        pred_jac[x_idx][z_idx_f] = z/denom
        pred_jac[u_idx][u_idx_f] = 1.0
        np.testing.assert_allclose(intermed_jac.toarray(), pred_jac)

        fcn_comp.set_input_values(inputs)
        jacobian = fcn_comp.evaluate_jacobian_outputs()
        pred_jac = np.zeros((2, 3))
        denom = np.sqrt(y**2 + z**2)
        pred_jac[0, y_idx_f] = 2*y
        pred_jac[0, z_idx_f] = 2*z
        pred_jac[0, u_idx_f] = 4*u
        pred_jac[1, y_idx_f] = u*y/denom
        pred_jac[1, z_idx_f] = u*z/denom
        pred_jac[1, u_idx_f] = denom
        np.testing.assert_allclose(jacobian.toarray(), pred_jac)

    def test_hessian(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        y, z, u = 2.0, 3.0, 4.0
        inputs = np.array([y, z, u])
        to_embed.set_input_values(inputs)
        intermed_hess = to_embed.evaluate_hessian_outputs()

        pred_hess = np.zeros((2, 3, 3))
        y_idx_f = to_embed.get_input_offset(0)
        z_idx_f = y_idx_f + 1
        u_idx_f = to_embed.get_input_offset(1)
        denom = np.sqrt(y**2 + z**2)
        pred_hess[x_idx, y_idx_f, y_idx_f] = z**2/denom**3
        pred_hess[x_idx, y_idx_f, z_idx_f] = -y*z/denom**3
        pred_hess[x_idx, z_idx_f, y_idx_f] = -y*z/denom**3
        pred_hess[x_idx, z_idx_f, z_idx_f] = y**2/denom**3
        # pred_hess[u_idx, ...] is all zeros

        for pred, act in zip(pred_hess, intermed_hess):
            np.testing.assert_allclose(pred, act.toarray())

        fcn_comp.set_input_values(inputs)
        hessian = fcn_comp.evaluate_hessian_outputs()
        pred_hess = np.zeros((2, 3, 3))
        pred_hess[0, y_idx_f, y_idx_f] = 2.0
        pred_hess[0, z_idx_f, z_idx_f] = 2.0
        pred_hess[0, u_idx_f, u_idx_f] = 4.0
        pred_hess[1, y_idx_f, y_idx_f] = u*z**2/denom**3
        pred_hess[1, y_idx_f, z_idx_f] = -u*z*y/denom**3
        pred_hess[1, z_idx_f, y_idx_f] = -u*z*y/denom**3
        pred_hess[1, z_idx_f, z_idx_f] = u*y**2/denom**3
        pred_hess[1, z_idx_f, u_idx_f] = z/denom
        pred_hess[1, u_idx_f, z_idx_f] = z/denom
        pred_hess[1, y_idx_f, u_idx_f] = y/denom
        pred_hess[1, u_idx_f, y_idx_f] = y/denom

        for pred, act in zip(pred_hess, hessian):
            # Need nonzero atol here because of numerical cancelation in the
            # calculation of the objective Hessian.
            np.testing.assert_allclose(pred, act.toarray(), atol=1e-15)


class TestNLPFromFunction(unittest.TestCase):

    def test_primals(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        nlp = NLPFromFunction(fcn_comp)
        self.assertEqual(nlp.n_primals(), 3)

    def test_constraints(self):
        m = make_model1_xu()
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        nlp = NLPFromFunction(fcn_comp)
        self.assertEqual(nlp.n_constraints(), 1)
        np.testing.assert_array_equal(nlp.constraints_lb(), [0.0])
        np.testing.assert_array_equal(nlp.constraints_ub(), [0.0])

    def test_cyipoptnlp(self):
        m = make_model1_yzu()
        nlp = PyomoNLP(m)
        problem = CyIpoptNLP(nlp)
        cyipopt = CyIpoptSolver(problem)
        cyipopt.solve(tee=True)

        m = make_model1_xu()
        pyomo_nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(pyomo_nlp)

        f1 = SimpleFunction2()
        f2 = IdentityFunction(1)
        functions = [None, None]
        x_idx, u_idx = pyomo_nlp.get_primal_indices([m.x, m.u])
        functions[x_idx] = f1
        functions[u_idx] = f2

        to_embed = FunctionStack(*functions)
        fcn_comp = FunctionComposition(nlp_fcn, to_embed)

        x0 = np.zeros(3)
        x_idx_f = to_embed.get_input_offset(x_idx)
        u_idx_f = to_embed.get_input_offset(u_idx)
        x0[x_idx_f] = 1.5
        x0[x_idx_f + 1] = 1.5
        x0[u_idx_f] = 2.5

        nlp = NLPFromFunction(fcn_comp)
        problem = CyIpoptNLP(nlp)
        cyipopt = CyIpoptSolver(problem)
        cyipopt.solve(x0=x0, tee=True)
        import pdb; pdb.set_trace()


class _TestReFixVars(unittest.TestCase):
    """
    This tests the functionality of using an NLP to "fix" variables
    by adding equality constraints.

    """

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2], initialize=5.0)
        m.u = pyo.Var(initialize=1.0)
        m.eq_con = pyo.Constraint(expr=m.x[1]*m.x[2] - m.u == 0.0)
        m.obj = pyo.Objective(expr=m.x[1]**2 + 3.0*m.x[2]**2)
        return m

    def _test_solves(self):
        m_fixed = self._make_model()
        m_fixed.u.fix(2.0)
        solver = pyo.SolverFactory("ipopt")
        solver.solve(m_fixed, tee=True)

        m = self._make_model()
        nlp = PyomoNLP(m)
        # Get index of variable(s) we would like to fix
        #

    def test_solve_constrained_model(self):
        m = self._make_model()
        m.fix_con = pyo.Constraint(expr=m.u - 2.0 == 0)
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        solver = pyo.SolverFactory("ipopt")
        solver.solve(m, tee=True)
        m.dual.pprint()

    def test_solve_constrained_model_as_nlp(self):
        m = self._make_model()
        m.fix_con = pyo.Constraint(expr=m.u - 2.0 == 0)
        nlp = PyomoNLP(m)
        nlp_fcn = FunctionFromNLP(nlp)
        fcn_nlp = NLPFromFunction(nlp_fcn)

        x0 = nlp.get_primals()
        fcn_nlp.set_primals(x0)

        problem = CyIpoptNLP(fcn_nlp)
        cyipopt = CyIpoptSolver(problem)
        x, results = cyipopt.solve(x0=x0, tee=True)
        import pdb; pdb.set_trace()

    def test_solve_fix_constraints(self):
        n_primals = 3
        value_map = {0: 1, 1: 2, 2: 3}
        fixing_constraints = FixedVarNLP(n_primals, value_map)

        problem = CyIpoptNLP(fixing_constraints)
        cyipopt = CyIpoptSolver(problem)
        x, results = cyipopt.solve(tee=True)
        import pdb; pdb.set_trace()

    def test_solve_fixed_nlp(self):
        m = self._make_model()
        nlp = PyomoNLP(m)
        pyomo_vars = [m.x[1], m.x[2], m.u]
        x1_idx, x2_idx, u_idx = nlp.get_primal_indices(pyomo_vars)
        n_primals = nlp.n_primals()
        value_map = {u_idx: 2.0}
        fixing_constraints = FixedVarNLP(n_primals, value_map)

        nlp_fcn = FunctionFromNLP(nlp)
        fixing_fcn = FunctionFromNLP(
            fixing_constraints, include_objective=False
        )

        combined_fcn = FunctionCombination(nlp_fcn, fixing_fcn)
        combined_nlp = NLPFromFunction(combined_fcn)

        x0 = nlp.get_primals()
        x0 = np.array([1.0, 1.0, 2.0])
        combined_nlp.set_primals(x0)

        problem = CyIpoptNLP(combined_nlp)
        cyipopt = CyIpoptSolver(problem)
        cyipopt.solve(x0=x0, tee=True)
        import pdb; pdb.set_trace()

    def test_fixed_nlp(self):
        m = self._make_model()
        nlp = PyomoNLP(m)
        pyomo_vars = [m.x[1], m.x[2], m.u]
        x1_idx, x2_idx, u_idx = nlp.get_primal_indices(pyomo_vars)
        n_primals = nlp.n_primals()
        value_map = {u_idx: 2.0}
        fixing_constraints = FixedVarNLP(n_primals, value_map)

        nlp_fcn = FunctionFromNLP(nlp)
        fixing_fcn = FunctionFromNLP(
            fixing_constraints, include_objective=False
        )

        combined_fcn = FunctionCombination(nlp_fcn, fixing_fcn)
        combined_nlp = NLPFromFunction(combined_fcn)
        self.assertEqual(combined_nlp.n_primals(), 3)
        self.assertEqual(combined_nlp.n_constraints(), 2)

        primals = nlp.get_primals()
        # This is necessary as I don't yet automatically set
        # init primals of an NLP-from-function
        combined_nlp.set_primals(primals)

        model_con_idx = nlp.get_constraint_indices([m.eq_con])[0]
        # I happen to know that the "fixing constraints" come after
        # the "model constraints"
        fix_con_idx = nlp.n_constraints()

        resid = combined_nlp.evaluate_constraints()
        pred_resid = np.zeros(2)
        pred_resid[model_con_idx] = pyo.value(m.eq_con.body)
        pred_resid[fix_con_idx] = m.u.value - 2.0
        np.testing.assert_allclose(resid, pred_resid)

        jac = combined_nlp.evaluate_jacobian()
        row = []
        col = []
        data = []

        row.append(model_con_idx)
        col.append(x1_idx)
        data.append(primals[x2_idx])

        row.append(model_con_idx)
        col.append(x2_idx)
        data.append(primals[x1_idx])

        row.append(model_con_idx)
        col.append(u_idx)
        data.append(-1.0)

        row.append(fix_con_idx)
        col.append(u_idx)
        data.append(1.0)

        pred_jac = sps.coo_matrix((data, (row, col)), shape=(2, 3))
        np.testing.assert_allclose(jac.toarray(), pred_jac.toarray())

        grad_obj = combined_nlp.evaluate_grad_objective()
        pred_grad = np.zeros(combined_nlp.n_primals())
        pred_grad[x1_idx] = 2.0*primals[x1_idx]
        pred_grad[x2_idx] = 6.0*primals[x2_idx]
        np.testing.assert_allclose(grad_obj, pred_grad)

        combined_nlp.set_duals(np.ones(combined_nlp.n_constraints()))
        hess = combined_nlp.evaluate_hessian_lag()
        row = []
        col = []
        data = []

        row.append(x1_idx)
        col.append(x2_idx)
        data.append(1.0)

        row.append(x2_idx)
        col.append(x1_idx)
        data.append(1.0)
        con_hess = sps.coo_matrix((data, (row, col)), shape=(3, 3))

        row = []
        col = []
        data = []
        row.append(x1_idx)
        col.append(x1_idx)
        data.append(2.0)

        row.append(x2_idx)
        col.append(x2_idx)
        data.append(6.0)
        obj_hess = sps.coo_matrix((data, (row, col)), shape=(3, 3))
        pred_hess = con_hess + obj_hess
        np.testing.assert_allclose(hess.toarray(), pred_hess.toarray())


if __name__ == '__main__':
    #unittest.main()
    _TestReFixVars().test_solve_constrained_model()
    _TestReFixVars().test_solve_fix_constraints()
    #_TestReFixVars().test_solve_constrained_model_as_nlp()
    #_TestReFixVars().test_solve_fixed_nlp()

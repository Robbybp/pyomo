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
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
    ExternalPyomoModel,
    get_hessian_of_constraint,
)
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
    ScalarExternalGreyBoxBlock,
    IndexedExternalGreyBoxBlock,
)
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
    PyomoNLPWithGreyBoxBlocks,
)
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
    PressureDropTwoOutputsWithHessian,
)

if not pyo.SolverFactory("ipopt").available():
    raise unittest.SkipTest(
        "Need IPOPT to run ExternalPyomoModel tests"
        )

"""
The following four functions define models used to test
the embedding of a scalar-valued function in an NLP.
"""

class SimpleModel(ExternalGreyBoxModel):
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
        h00 = (self.outputs[0]**2 - self.inputs[0]**2)/self.outputs[0]**3
        h01 = - self.inputs[0]*self.inputs[1]/self.outputs[0]**3
        h11 = (self.outputs[0]**2 - self.inputs[1]**2)/self.outputs[0]**3
        self.hessian_outputs[0, 0, 0] = h00
        self.hessian_outputs[0, 1, 0] = h01
        self.hessian_outputs[0, 0, 1] = h10
        self.hessian_outputs[0, 1, 1] = h11

    def n_output(self):
        return len(self.outputs)

    def evaluate_outputs(self):
        return self.outputs

    def evaluate_jacobian_outputs(self):
        return sps.coo_matrix(self.jacobian_outputs)

    def evaluate_hessian_outputs(self):
        # TODO: What data format should I use for sparse tensors?
        # Want:
        # - fast access along rank 1 (to get an individual output's hessian matrix)
        # - Fast multiplication and transpose for a slice across rank 1
        # - Identification of a slice across ranks 2 and 3. This is what
        #   an array of coo matrices doesn't handle well...
        return sps.coo_matrix()

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


if __name__ == '__main__':
    unittest.main()

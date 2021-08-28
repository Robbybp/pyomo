#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from pyomo.contrib.pynumero.interfaces.nlp import NLP
import numpy as np
import scipy.sparse as sps


class FixedVarNLP(NLP):
    """
    This is an experimental NLP that supplies equality constraints
    necessary to fix specified variables.
    """

    def __init__(self, n_primals, value_map):
        self._n_primals = n_primals
        self._value_map = dict(value_map) # Copy the user's dict
        for i in value_map:
            assert 0 <= i and i < n_primals
        self._fixed_primal_coords = np.array([
            i for i in range(n_primals) if i in value_map
        ])
        self._fixed_primal_values = np.array([
            value_map[i] for i in range(n_primals) if i in value_map
        ])
        n_constraints = len(value_map)
        self._n_constraints = n_constraints
        self._primals_lb = np.array([-np.inf for _ in range(n_primals)])
        self._primals_ub = np.array([np.inf for _ in range(n_primals)])
        self._constraints_lb = np.array([0.0 for _ in range(n_constraints)])
        self._constraints_ub = np.array([0.0 for _ in range(n_constraints)])
        self._obj_factor = 1.0

        self._primals = self.init_primals()
        self._duals = self.init_duals()

    def update_fixed_values(self, value_map):
        assert all(i in self._value_map for i in value_map)
        for i, newval in value_map.items():
            self._value_map[i] = newval
    
    def n_primals(self):
        return self._n_primals

    def n_constraints(self):
        return self._n_constraints

    def nnz_jacobian(self):
        return self._n_constraints

    def nnz_hessian_lag(self):
        return 0

    def primals_lb(self):
        return self._primals_lb

    def primals_ub(self):
        return self._primals_ub

    def constraints_lb(self):
        return self._constraints_lb

    def constraints_ub(self):
        return self._constraints_ub

    def init_primals(self):
        return np.zeros(self.n_primals())

    def init_duals(self):
        return np.zeros(self.n_primals())

    def create_new_vector(self, vector_type):
        raise NotImplementedError()

    def set_primals(self, primals):
        np.copyto(self._primals, primals)

    def get_primals(self):
        return self._primals.copy()

    def set_duals(self, duals):
        np.copyto(self._duals, duals)

    def get_duals(self):
        return self._duals.copy()

    def set_obj_factor(self, obj_factor):
        self._obj_factor = obj_factor

    def get_obj_factor(self):
        return self._obj_factor

    def get_obj_scaling(self):
        return 1.0

    def get_primals_scaling(self):
        return np.ones(self.n_primals())

    def get_constraints_scaling(self):
        return np.ones(self.n_constraints())

    def evaluate_objective(self):
        return 0.0

    def evaluate_grad_objective(self, out=None):
        return np.zeros(self.n_primals())

    def evaluate_constraints(self, out=None):
        N = self.n_primals()
        residuals = np.array([
            self._primals[i] - self._value_map[i]
            for i in range(N) if i in self._value_map
        ])
        return residuals

    def evaluate_jacobian(self, out=None):
        N = self.n_primals()
        M = self.n_constraints()
        row = list(range(M))
        # TODO: Numpy array and compress?
        col = [i for i in range(N) if i in self._value_map]
        data = [1.0 for i in range(N) if i in self._value_map]
        return sps.coo_matrix((data, (row, col)), shape=(M, N))

    def evaluate_hessian_lag(self, out=None):
        N = self.n_primals()
        return sps.coo_matrix((N, N))

    def report_solver_status(self, status_code, status_message):
        raise NotImplementedError()

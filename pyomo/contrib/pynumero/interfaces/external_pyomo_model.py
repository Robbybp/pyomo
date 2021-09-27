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
from pyomo.environ import SolverFactory, TerminationCondition
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.objective import Objective
from pyomo.core.base.reference import Reference
from pyomo.core.expr.visitor import (
        identify_variables,
        identify_mutable_parameters,
        )
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.subsystems import (
        create_subsystem_block,
        TemporarySubsystemManager,
        )
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxModel,
        )
from pyomo.contrib.pynumero.interfaces.functions import (
        NLPFromFunction,
        FunctionFromNLP,
        FunctionCombination,
        )
from pyomo.contrib.pynumero.interfaces.abstract_nlps import (
        FixedVarNLP,
        )
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
        CyIpoptNLP,
        CyIpoptSolver,
        cyipopt_available,
        )
from pyomo.contrib.incidence_analysis.util import (
        generate_strongly_connected_components,
        )
import numpy as np
import scipy.sparse as sps


class ImplicitFunctionError(RuntimeError):
    pass


def _dense_to_full_sparse(matrix):
    """
    Used to convert a dense matrix (2d NumPy array) to SciPy sparse matrix
    with explicit coordinates for every entry, including zeros. This is
    used because _ExternalGreyBoxAsNLP methods rely on receiving sparse
    matrices where sparsity structure does not change between calls.
    This is difficult to achieve for matrices obtained via the implicit
    function theorem unless an entry is returned for every coordinate
    of the matrix.

    Note that this does not mean that the Hessian of the entire NLP will
    be dense, only that the block corresponding to this external model
    will be dense.
    """
    # TODO: Allow methods to hard-code Jacobian/Hessian sparsity structure
    # in the case it is known a priori.
    # TODO: Decompose matrices to infer maximum fill-in sparsity structure.
    nrow, ncol = matrix.shape
    row = []
    col = []
    data = []
    for i, j in itertools.product(range(nrow), range(ncol)):
        row.append(i)
        col.append(j)
        data.append(matrix[i,j])
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    return sps.coo_matrix((data, (row, col)), shape=(nrow, ncol))


def get_hessian_of_constraint(constraint, wrt1=None, wrt2=None, nlp=None):
    constraints = [constraint]
    if wrt1 is None and wrt2 is None:
        variables = list(identify_variables(constraint.expr, include_fixed=False))
        wrt1 = variables
        wrt2 = variables
    elif wrt1 is not None and wrt2 is not None:
        variables = wrt1 + wrt2
    elif wrt1 is not None: # but wrt2 is None
        wrt2 = wrt1
        variables = wrt1
    else:
        # wrt2 is not None and wrt1 is None
        wrt1 = wrt2
        variables = wrt1

    if nlp is None:
        block = create_subsystem_block(constraints, variables=variables)
        # Could fix input_vars so I don't evaluate the Hessian with respect
        # to variables I don't care about...

        # HUGE HACK: Variables not included in a constraint are not written
        # to the nl file, so we cannot take the derivative with respect to
        # them, even though we know this derivative is zero. To work around,
        # we make sure all variables appear on the block in the form of a
        # dummy constraint. Then we can take derivatives of any constraint
        # with respect to them. Conveniently, the extract_submatrix_
        # call deals with extracting the variables and constraint we care
        # about, in the proper order.
        block._dummy_var = Var()
        block._dummy_con = Constraint(expr=sum(variables) == block._dummy_var)
        block._obj = Objective(expr=0.0)
        nlp = PyomoNLP(block)

    saved_duals = nlp.get_duals()
    saved_obj_factor = nlp.get_obj_factor()
    temp_duals = np.zeros(len(saved_duals))

    # NOTE: This makes some assumption about how the Lagrangian is constructed.
    # TODO: Define the convention we assume and convert if necessary.
    idx = nlp.get_constraint_indices(constraints)[0]
    temp_duals[idx] = 1.0
    nlp.set_duals(temp_duals)
    nlp.set_obj_factor(0.0)

    # NOTE: The returned matrix preserves explicit zeros. I.e. it contains
    # coordinates for every entry that could possibly be nonzero.
    submatrix = nlp.extract_submatrix_hessian_lag(wrt1, wrt2)

    nlp.set_obj_factor(saved_obj_factor)
    nlp.set_duals(saved_duals)
    return submatrix


class ExternalPyomoModel(ExternalGreyBoxModel):
    """
    This is an ExternalGreyBoxModel used to create an external model
    from existing Pyomo components. Given a system of variables and
    equations partitioned into "input" and "external" variables and
    "residual" and "external" equations, this class computes the
    residual of the "residual equations," as well as their Jacobian
    and Hessian, as a function of only the inputs.

    Pyomo components:
        f(x, y) == 0 # "Residual equations"
        g(x, y) == 0 # "External equations", dim(g) == dim(y)

    Effective constraint seen by this "external model":
        F(x) == f(x, y(x)) == 0
        where y(x) solves g(x, y) == 0

    """

    def __init__(self,
            input_vars,
            external_vars,
            residual_cons,
            external_cons,
            solver=None,
            decompose=False,
            ):
        # We only need this block to construct the NLP, which wouldn't
        # be necessary if we could compute Hessians of Pyomo constraints.
        self._block = create_subsystem_block(
                residual_cons+external_cons,
                input_vars+external_vars,
                )
        self._block._obj = Objective(expr=0.0)
        self._nlp = PyomoNLP(self._block)

        self._external_block = create_subsystem_block(
            external_cons, external_vars
        )
        self._external_block._obj = Objective(expr=0.0)

        self._use_cyipopt = False
        self._n_external_nlps = 1
        if cyipopt_available and solver is None and decompose:
            # (a) What data do I need to solve a subsystem?
            # (b) Is any of this data being used elsewhere?
            #     - NLPs of external constraints - no
            #     - coordinates in the NLP belonging to inputs - no
            #       these can get updated as soon as set_inputs is called
            #     - subset of inputs that appear in the NLP - no
            #       ^ Can inputs coords/NLP coords be combined in a map?
            #       With above map, it is trivial to get values to set
            #       (However, also need to set external var values)
            #       (and combine into a single map to send to fixing NLP)
            #       (each NLP potentially needs to receive data from all
            #       previous NLPs)
            #     - fixing constraints for each nlp - not used elsewhere
            #       Needs to fix "global" inputs and "local" inputs...
            #     - Initialize inputs in the actual external NLP... 
            self._use_cyipopt = True

            # Blocks in the SCC decomposition of our external system
            self._external_block_decomp = []
            # "Inputs" into each block of decomposition that we must fix
            decomp_to_fix = []
            for i, (block, inputs) in enumerate(
                        generate_strongly_connected_components(
                        list(self._external_block.cons.values()),
                        list(self._external_block.vars.values()),
                    )):
                block._obj = Objective(expr=0.0)
                self._external_block_decomp.append(block)
                decomp_to_fix.append(inputs)

            # Temporarily hard-code the full system
            #self._external_block_decomp = [self._external_block]
            #decomp_to_fix = [list(self._external_block.input_vars.values())]
            #

            # NLP for each block of decomposition
            self._external_nlps = [
                PyomoNLP(block) for block in self._external_block_decomp
            ]
            self._n_external_nlps = len(self._external_nlps)

            # All the variables we need to fix in the external system.
            # input_vars may be only a subset of these, and these may
            # include only a subset of input_vars
            #to_fix = list(self._external_block.input_vars.values())
            #to_fix_set = ComponentSet(to_fix)
            decomp_to_fix_sets = [
                ComponentSet(inputs) for inputs in decomp_to_fix
            ]

            # Inputs that participate in the external system
            # Do we need to know about "global inputs" for each
            # block in decomposition? Yes, probably, so we can fix
            # these as soon as the values become available.
            self._inputs_in_each_external = [
                [i for i, var in enumerate(input_vars) if var in set_]
                for set_ in decomp_to_fix_sets
            ]
            # The actual input variable objects. Need these to get the
            # coordinates in the external nlps
            inputs_to_fix = [
                [var for var in input_vars if var in set_]
                for set_ in decomp_to_fix_sets
            ]

            # Coordinates in external systems belonging to "global" inputs
            input_coords_external = [
                nlp.get_primal_indices(inputs)
                for nlp, inputs in zip(self._external_nlps, inputs_to_fix)
            ]
            self._input_coords_external = input_coords_external

            # Set values of variables to "fix" in the external system
            value_maps = [
                dict(zip(
                    nlp.get_primal_indices(to_fix), [v.value for v in to_fix]
                ))
                for nlp, to_fix in zip(self._external_nlps, decomp_to_fix)
            ]
            # NLPs that fix "global inputs" in each external system
            self._fixing_constraints = [
                FixedVarNLP(nlp.n_primals(), value_map)
                for nlp, value_map in zip(self._external_nlps, value_maps)
            ]
            # Combine external NLP with "fixing constraints" NLP
            nlp_fcns = [FunctionFromNLP(nlp) for nlp in self._external_nlps]
            fixing_fcns = [
                FunctionFromNLP(cons, include_objective=False)
                for cons in self._fixing_constraints
            ]
            # TODO: For each external NLP, get the NLPs required to fix
            # "local inputs" from all previous NLPs.

            # What data do I need for this implementation?
            # self._prev_nlp_coords
            # self._curr_nlp_coords
            # self._previous_var_constraints
            self._prev_nlp_coords = [
                [None for j in range(i)] for i in range(self._n_external_nlps)
            ]
            self._curr_nlp_coords = [
                [None for j in range(i)] for i in range(self._n_external_nlps)
            ]
            self._previous_var_constraints = []
            for i in range(self._n_external_nlps):
                nlp_i = self._external_nlps[i]
                # Variables not in the subsystem
                extra_vars = decomp_to_fix_sets[i]
                value_map = {}
                for j in range(i):
                    block_j = self._external_block_decomp[j]
                    nlp_j = self._external_nlps[j]
                    vars_j = list(block_j.vars.values())
                    prev_vars = [var for var in vars_j if var in extra_vars]
                    prev_coords = nlp_j.get_primal_indices(prev_vars)
                    curr_coords = nlp_i.get_primal_indices(prev_vars)
                    self._prev_nlp_coords[i][j] = prev_coords
                    self._curr_nlp_coords[i][j] = curr_coords

                    #values = [v.value for v in vars_j]
                    #value_map.update(dict(zip(curr_coords, values)))

                #self._previous_var_constraints.append(
                #    FixedVarNLP(nlp_i.n_primals(), value_map)
                #)

            #previous_var_fcns = [
            #    FunctionFromNLP(cons, include_objective=False)
            #    for cons in self._previous_var_constraints
            #]

            combined_fcns = [
                FunctionCombination(nlp_fcn, fixing_fcn)
                for nlp_fcn, fixing_fcn in zip(nlp_fcns, fixing_fcns)
            ]
            #combined_fcns = [
            #    FunctionCombination(nlp_fcn, fixing_fcn, prev_fcn)
            #    for nlp_fcn, fixing_fcn, prev_fcn in
            #    zip(nlp_fcns, fixing_fcns, previous_var_fcns)
            #]
            combined_nlps = [NLPFromFunction(fcn) for fcn in combined_fcns]
            problems = [CyIpoptNLP(nlp) for nlp in combined_nlps]
            self._solvers = [CyIpoptSolver(problem) for problem in problems]
        elif solver is None:
            self._solver = SolverFactory("ipopt")
        else:
            self._solver = solver

        assert len(external_vars) == len(external_cons)

        self.input_vars = input_vars
        self.external_vars = external_vars
        self.residual_cons = residual_cons
        self.external_cons = external_cons

        self.residual_con_multipliers = [None for _ in residual_cons]
        self.residual_scaling_factors = None

    def n_inputs(self):
        return len(self.input_vars)

    def n_equality_constraints(self):
        return len(self.residual_cons)

    # I would like to try to get by without using the following "name" methods.
    def input_names(self):
        return ["input_%i" % i for i in range(self.n_inputs())]
    def equality_constraint_names(self):
        return ["residual_%i" % i for i in range(self.n_equality_constraints())]

    def set_input_values(self, input_values):
        external_cons = self.external_cons
        external_vars = self.external_vars
        external_block = self._external_block
        input_vars = self.input_vars

        for var, val in zip(input_vars, input_values):
            var.set_value(val)

        if self._use_cyipopt:
            # No solver for the implicit function system was provided.
            # We default to solving with CyIpopt as we have a direct
            # interface that lets us avoid writing an nl file each
            # iteration.

            # - Set "global" input values for all NLPs
            #   (This is basically what I'm doing right now)
            # - Iterate over NLPs, solve
            # - Update values in future/past NLPs
            # - Update Pyomo variables from each NLP

            # Update "global" inputs
            for i in range(self._n_external_nlps):
                input_coords = self._inputs_in_each_external[i]
                input_value_array = np.array(input_values)[input_coords]
                input_coords_in_nlp = self._input_coords_external[i]
                value_map = dict(zip(input_coords_in_nlp, input_value_array))
                fixing_constraints = self._fixing_constraints[i]
                fixing_constraints.update_fixed_values(value_map)

                # Initialize variable values
                external_primals = self._external_nlps[i].get_primals()
                external_primals[input_coords_in_nlp] = input_value_array
                nlp = self._external_nlps[i]
                nlp.set_primals(external_primals)

            # TODO: update "local" inputs
            for i in range(self._n_external_nlps):
                nlp = self._external_nlps[i]
                primals = nlp.get_primals()
                # Coords in external systems that were solved for in a
                # previous external system.
                value_map = {}
                for j in range(i):
                    prev_nlp = self._external_nlps[j]
                    # Coords of this NLP that appear in NLP i
                    coords = self._prev_nlp_coords[i][j]
                    prev_primals = prev_nlp.get_primals()
                    # Need to map these values to their coord in NLP i
                    values = prev_primals[coords]
                    nlp_coords = self._curr_nlp_coords[i][j]
                    value_map.update(dict(zip(nlp_coords, values)))
                fixing_constraints = self._fixing_constraints[i]
                fixing_constraints.update_fixed_values(value_map)
                for idx, val in value_map.items():
                    # TODO: Do this with vectorized syntax
                    primals[idx] = val
                #primals[prev_var_coords] = prev_var_values
                nlp.set_primals(primals)

                # Solve each NLP
                external_primals = self._external_nlps[i].get_primals()
                solver = self._solvers[i]
                x0 = external_primals
                x, res = solver.solve(x0=x0)
                if res["status"] != 0:
                    raise ImplicitFunctionError(
                        "Failed to properly converge implicit function "
                        "subproblem with solver of type %s.\n"
                        "Message from the solver is: %s"
                        % (type(solver), res["status_msg"])
                    )

                # Update Pyomo values after solve.
                pyomo_vars = self._external_nlps[i].get_pyomo_variables()
                assert len(x) == len(pyomo_vars)
                for var, val in zip(pyomo_vars, x):
                    var.set_value(val)

            # TODO: solve via decomposition here.
            # The partition should be computed in __init__, then
            # here we solve a sequence of square problems (with
            # a sequence of CyIpoptNLPs).
            # We should have the option to perform this decomposition or not.
            # If we don't, our sequence of subproblems should just be length one.
            #
            # external_nlp -> external_nlps
            #
            # (i)   Put data I need in lists, iterate over these len-1 lists here
            # (ii)  Do the decomposition in __init__
            # (iii) Here, perform the necessary update between adjacent solves
            # (iv)  Populate lists with data from decomposition

            # Compress provided input values to keep those in the external
            # system.
            # These are coords in the NLP
            #input_coords = self._input_coords_external
            #input_value_array = np.array(input_values)[self._inputs_in_external]
            ## Update fixed values of input variables in the external system
            #value_map = dict(zip(input_coords, input_value_array))
            #self._fixing_constraints.update_fixed_values(value_map)

            ## Update current values of fixed variables
            #external_primals[input_coords] = input_value_array
            #self._external_nlp.set_primals(external_primals)

            ## Set initial guess and solve with CyIpopt
            #x0 = external_primals
            #x, res = solver.solve(x0=x0)
            #if res["status"] != 0:
            #    raise ImplicitFunctionError(
            #        "Failed to properly converge implicit function "
            #        "subproblem with solver of type %s.\n"
            #        "Message from the solver is: %s"
            #        % (type(solver), res["status_msg"])
            #    )

            ## Update Pyomo values after solve.
            ##
            ## We will still update variable values after each solve.
            ##
            #pyomo_vars = self._external_nlp.get_pyomo_variables()
            #for var, val in zip(pyomo_vars, x):
            #    var.set_value(val)

        else:
            # CyIpopt is unavailable or a solver was provided, so we
            # solve the Pyomo model of the implicit function system.
            #
            # DECOMP: I should create a sequence of external blocks
            # in addition to NLPs.
            #
            solver = self._solver
            block = self._external_block
            with TemporarySubsystemManager(
                    to_fix=list(block.input_vars.values())
                    ):
                for scc, inputs in generate_strongly_connected_components(
                        list(block.cons.values()),
                        list(block.vars.values()),
                        ):
                    with TemporarySubsystemManager(to_fix=inputs):
                        if len(scc.vars) == 1:
                            calculate_variable_from_constraint(
                                scc.vars[0],
                                scc.cons[0],
                            )
                        else:
                            res = solver.solve(scc)
                            if (res.solver.termination_condition is not
                                    TerminationCondition.optimal):
                                raise ImplicitFunctionError(
                                    "Failed to properly converge implicit function "
                                    "subproblem with solver of type %s.\n"
                                    "Message from the solver is: %s"
                                    % (type(solver), res.solver.message)
                                )

        # DECOMP: This code should still be valid. Pyomo vars should be
        # properly updated.
        input_value_map = ComponentMap(zip(input_vars, input_values))
        external_value_map = ComponentMap(
            ((var, var.value) for var in external_vars)
        )

        primal_vars = self._nlp.get_pyomo_variables()
        #primals = self._nlp.get_primals()
        #for i, var in enumerate(primal_vars):
        #    if var in input_value_map:
        #        primals[i] = input_value_map[var]
        #    elif var in external_value_map:
        #        primals[i] = external_value_map[var]
        primals = np.array([v.value for v in primal_vars])
        self._nlp.set_primals(primals)

    def set_equality_constraint_multipliers(self, eq_con_multipliers):
        eq_con_multipliers = np.array(eq_con_multipliers)
        for i, val in enumerate(eq_con_multipliers):
            self.residual_con_multipliers[i] = val
        external_multipliers = self.calculate_external_constraint_multipliers(
            eq_con_multipliers,
        )
        multipliers = np.concatenate((eq_con_multipliers, external_multipliers))
        cons = self.residual_cons + self.external_cons
        n_con = len(cons)
        assert n_con == self._nlp.n_constraints()
        duals = np.zeros(n_con)
        indices = self._nlp.get_constraint_indices(cons)
        for i, idx in enumerate(indices):
            duals[idx] = multipliers[i]
        self._nlp.set_duals(duals)

    def calculate_external_constraint_multipliers(self, resid_multipliers):
        nlp = self._nlp
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfy = nlp.extract_submatrix_jacobian(y, f)
        jgy = nlp.extract_submatrix_jacobian(y, g)

        jgy_t = jgy.transpose()
        jfy_t = jfy.transpose()
        dfdg = - sps.linalg.splu(jgy_t.tocsc()).solve(jfy_t.toarray())
        resid_multipliers = np.array(resid_multipliers)
        external_multipliers = dfdg.dot(resid_multipliers)
        return external_multipliers

    def get_hessians_of_lagrangian(self):
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        hlxx = nlp.extract_submatrix_hessian_lag(x, x)
        hlxy = nlp.extract_submatrix_hessian_lag(x, y)
        hlyy = nlp.extract_submatrix_hessian_lag(y, y)
        return hlxx, hlxy, hlyy

    def calculate_reduced_hessian_lagrangian(self, hlxx, hlxy, hlyy):
        hlxx = hlxx.toarray()
        hlxy = hlxy.toarray()
        hlyy = hlyy.toarray()
        dydx = self.evaluate_jacobian_external_variables()
        term1 = hlxx
        prod = hlxy.dot(dydx)
        term2 = prod + prod.transpose()
        term3 = hlyy.dot(dydx).transpose().dot(dydx)
        hess_lag = term1 + term2 + term3
        return hess_lag

    def evaluate_equality_constraints(self):
        return self._nlp.extract_subvector_constraints(self.residual_cons)

    def evaluate_jacobian_equality_constraints(self):
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfx = nlp.extract_submatrix_jacobian(x, f)
        jfy = nlp.extract_submatrix_jacobian(y, f)
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)

        nf = len(f)
        nx = len(x)
        n_entries = nf*nx

        # TODO: Does it make sense to cast dydx to a sparse matrix?
        # My intuition is that it does only if jgy is "decomposable"
        # in the strongly connected component sense, which is probably
        # not usually the case.
        dydx = -1 * sps.linalg.splu(jgy.tocsc()).solve(jgx.toarray())
        # NOTE: PyNumero block matrices require this to be a sparse matrix
        # that contains coordinates for every entry that could possibly
        # be nonzero. Here, this is all of the entries.
        dfdx = jfx + jfy.dot(dydx)

        return _dense_to_full_sparse(dfdx)

    def evaluate_jacobian_external_variables(self):
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        g = self.external_cons
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)
        jgy_csc = jgy.tocsc()
        dydx = -1 * sps.linalg.splu(jgy_csc).solve(jgx.toarray())
        return dydx

    def evaluate_hessian_external_variables(self):
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        g = self.external_cons
        jgx = nlp.extract_submatrix_jacobian(x, g)
        jgy = nlp.extract_submatrix_jacobian(y, g)
        jgy_csc = jgy.tocsc()
        jgy_fact = sps.linalg.splu(jgy_csc)
        dydx = -1 * jgy_fact.solve(jgx.toarray())

        ny = len(y)
        nx = len(x)

        hgxx = np.array([
            get_hessian_of_constraint(con, x, nlp=nlp).toarray() for con in g
            ])
        hgxy = np.array([
            get_hessian_of_constraint(con, x, y, nlp=nlp).toarray() for con in g
            ])
        hgyy = np.array([
            get_hessian_of_constraint(con, y, nlp=nlp).toarray() for con in g
            ])

        # This term is sparse, but we do not exploit it.
        term1 = hgxx

        # This is what we want.
        # prod[i,j,k] = sum(hgxy[i,:,j] * dydx[:,k])
        prod = hgxy.dot(dydx)
        # Swap the second and third axes of the tensor
        term2 = prod + prod.transpose((0, 2, 1))
        # The term2 tensor could have some sparsity worth exploiting.

        # matrix.dot(tensor) is not what we want, so we reverse the order of the
        # product. Exploit symmetry of hgyy to only perform one transpose.
        term3 = hgyy.dot(dydx).transpose((0, 2, 1)).dot(dydx)

        rhs = term1 + term2 + term3

        rhs.shape = (ny, nx*nx)
        sol = jgy_fact.solve(rhs)
        sol.shape = (ny, nx, nx)
        d2ydx2 = -sol

        return d2ydx2

    def evaluate_hessians_of_residuals(self):
        """
        This method computes the Hessian matrix of each equality
        constraint individually, rather than the sum of Hessians
        times multipliers.
        """
        nlp = self._nlp
        x = self.input_vars
        y = self.external_vars
        f = self.residual_cons
        g = self.external_cons
        jfx = nlp.extract_submatrix_jacobian(x, f)
        jfy = nlp.extract_submatrix_jacobian(y, f)

        dydx = self.evaluate_jacobian_external_variables()

        ny = len(y)
        nf = len(f)
        nx = len(x)

        hfxx = np.array([
            get_hessian_of_constraint(con, x, nlp=nlp).toarray() for con in f
            ])
        hfxy = np.array([
            get_hessian_of_constraint(con, x, y, nlp=nlp).toarray() for con in f
            ])
        hfyy = np.array([
            get_hessian_of_constraint(con, y, nlp=nlp).toarray() for con in f
            ])

        d2ydx2 = self.evaluate_hessian_external_variables()

        term1 = hfxx
        prod = hfxy.dot(dydx)
        term2 = prod + prod.transpose((0, 2, 1))
        term3 = hfyy.dot(dydx).transpose((0, 2, 1)).dot(dydx)

        d2ydx2.shape = (ny, nx*nx)
        term4 = jfy.dot(d2ydx2)
        term4.shape = (nf, nx, nx)

        d2fdx2 = term1 + term2 + term3 + term4
        return d2fdx2

    def _evaluate_hessian_equality_constraints(self):
        """
        This method actually evaluates the sum of Hessians times
        multipliers, i.e. the term in the Hessian of the Lagrangian
        due to these equality constraints.
        """
        d2fdx2 = self.evaluate_hessians_of_residuals()
        multipliers = self.residual_con_multipliers

        sum_ = sum(mult*matrix for mult, matrix in zip(multipliers, d2fdx2))
        # Return a sparse matrix with every entry accounted for because it
        # is difficult to determine rigorously which coordinates
        # _could possibly_ be nonzero.
        sparse = _dense_to_full_sparse(sum_)
        return sps.tril(sparse)

    def set_equality_constraint_scaling_factors(self, scaling_factors):
        """
        """
        self.residual_scaling_factors = np.array(scaling_factors)

    def get_equality_constraint_scaling_factors(self):
        """
        """
        return self.residual_scaling_factors

    def evaluate_hessian_equality_constraints(self):
        """
        """
        hlxx, hlxy, hlyy = self.get_hessians_of_lagrangian()
        hess_lag = self.calculate_reduced_hessian_lagrangian(hlxx, hlxy, hlyy)
        sparse = _dense_to_full_sparse(hess_lag)
        return sps.tril(sparse)

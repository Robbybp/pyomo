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

from collections import namedtuple

from pyomo.common.collections import ComponentSet
from pyomo.common.timing import HierarchicalTimer
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.core.base.objective import Objective
from pyomo.core.base.suffix import Suffix
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.contrib.pynumero.interfaces.nlp_projections import ProjectedNLP
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import (
    cyipopt_available,
    CyIpoptNLP,
    CyIpoptSolver,
)
from pyomo.contrib.pynumero.algorithms.solvers.square_solver_base import (
    ParameterizedSquareSolver,
)
from pyomo.contrib.incidence_analysis.util import (
    generate_strongly_connected_components,
)


class SccMultiNlpHybridParamSquareSolver(ParameterizedSquareSolver):

    TimeBins = namedtuple(
        "TimeBins",
        [
            "construct",
            "update_parameters",
            "solve",
            "calc_var",
            "cyipopt",
        ],
    )
    time_bins = TimeBins(
        *ParameterizedSquareSolver.time_bins,
        "calc_var",
        "newton",
    )

    def __init__(
        self,
        model,
        param_vars,
        variables=None,
        timer=None,
    ):
        """
        Arguments
        ---------
        model: Block
            Block to be solved
        param_vars: List of VarData
            Variables to be treated as parameters
        variables: List of VarData
            Variables to be solved for

        """
        self._model = model
        self._param_vars = param_vars
        self._param_var_set = ComponentSet(param_vars)
        if timer is None:
            timer = HierarchicalTimer()
        self._timer = timer

        self._timer.start(self.time_bins.construct)

        self.equations = list(
            model.component_data_objects(Constraint, active=True)
        )
        if variables is None:
            variables = [
                v for v in model.component_data_objects(Var)
                if not v.fixed and not v in self._param_var_set
            ]
        self.variables = variables
        if len(self.variables) != len(self.equations):
            raise RuntimeError()

        self._scc_list = list(generate_strongly_connected_components(
            self.equations, variables=self.variables
        ))
        self._vector_scc_list = [
            (scc, inputs) for scc, inputs in self._scc_list
            if len(scc.vars) > 1
        ]

        # Need a dummy objective to create an NLP
        for scc, inputs in self._vector_scc_list:
            scc._obj = Objective(expr=0.0)

            # I need scaling_factor so Pyomo NLPs I create from these blocks
            # don't break when ProjectedNLP calls get_primals_scaling
            scc.scaling_factor = Suffix(direction=Suffix.EXPORT)
            # HACK: scaling_factor just needs to be nonempty.
            scc.scaling_factor[scc._obj] = 1.0

        # These are the "original NLPs" that will be projected
        self._vector_scc_nlps = [
            PyomoNLP(scc) for scc, inputs in self._vector_scc_list
        ]
        self._vector_scc_var_names = [
            [var.name for var in scc.vars.values()]
            for scc, inputs in self._vector_scc_list
        ]
        self._vector_proj_nlps = [
            ProjectedNLP(nlp, names) for nlp, names in
            zip(self._vector_scc_nlps, self._vector_scc_var_names)
        ]

        # We will solve the ProjectedNLPs rather than the original NLPs
        self._cyipopt_nlps = [CyIpoptNLP(nlp) for nlp in self._vector_proj_nlps]
        self._cyipopt_solvers = [
            CyIpoptSolver(nlp) for nlp in self._cyipopt_nlps
        ]
        self._vector_scc_input_coords = [
            nlp.get_primal_indices(inputs)
            for nlp, (scc, inputs) in
            zip(self._vector_scc_nlps, self._vector_scc_list)
        ]

        self._timer.stop(self.time_bins.construct)

    def update_parameters(self, values):
        self._timer.start(self.time_bins.update_parameters)
        for var, val in zip(self._param_vars, values):
            var.set_value(val, skip_validation=True)
        self._timer.stop(self.time_bins.update_parameters)

    def solve(self):
        self._timer.start(self.time_bins.solve)
        vector_scc_idx = 0
        for block, inputs in self._scc_list:
            if len(block.vars) == 1:
                self._timer.start(self.time_bins.calc_var)
                calculate_variable_from_constraint(
                    block.vars[0], block.cons[0]
                )
                self._timer.stop(self.time_bins.calc_var)
            else:
                # Transfer variable values into the projected NLP, solve,
                # and extract values.
                self._timer.start(self.time_bins.calc_var)
                self._timer.stop(self.time_bins.calc_var)

                nlp = self._vector_scc_nlps[vector_scc_idx]
                proj_nlp = self._vector_proj_nlps[vector_scc_idx]
                input_coords = self._vector_scc_input_coords[vector_scc_idx]
                cyipopt = self._cyipopt_solvers[vector_scc_idx]
                _, local_inputs = self._vector_scc_list[vector_scc_idx]

                primals = nlp.get_primals()
                variables = nlp.get_pyomo_variables()

                # Set values and bounds from inputs to the SCC.
                # This works because values have been set in the original
                # pyomo model, either by a previous SCC solve, or from the
                # "global inputs"
                for i, var in zip(input_coords, local_inputs):
                    # Set primals (inputs) in the original NLP
                    primals[i] = var.value
                # This affects future evaluations in the ProjectedNLP

                nlp.set_primals(primals)

                x0 = proj_nlp.get_primals()

                self._timer.start(self.time_bins.cyipopt)
                sol, _ = cyipopt.solve(x0=x0)
                self._timer.stop(self.time_bins.cyipopt)

                # Set primals from solution in projected NLP. This updates
                # values in the original NLP
                proj_nlp.set_primals(sol)

                # I really only need to set new primals for the variables in
                # the ProjectedNLP. However, I can only get a list of variables
                # from the original Pyomo NLP, so here some of the values I'm
                # setting are redundant.
                new_primals = nlp.get_primals()
                assert len(new_primals) == len(variables)
                for var, val in zip(variables, new_primals):
                    var.set_value(val, skip_validation=True)

                vector_scc_idx += 1
        self._timer.stop(self.time_bins.solve)

from collections import namedtuple


TimeBins = namedtuple(
    "TimeBins",
    ["construct", "update_parameters", "solve"],
)


class ParameterizedSquareSolver(object):
    """
    Given a square Pyomo model representing a system:
    g(x, y) = 0
    where x are variables and y are parameters.
    This class allows updating parameters and solving for variables.

    """

    time_bins = TimeBins("construct", "update_parameters", "solve")

    def __init__(self, model, param_vars):
        # Is param_vars a confusing name? These must be variables, but they
        # will be treated as paramters
        # In a proper Pyomo interface, these could be actual parameters
        # or variables. Maybe this should even go through the existing
        # sensitivity interface.
        self._model = model
        self._param_vars = param_vars

    def update_parameters(self, values):
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()

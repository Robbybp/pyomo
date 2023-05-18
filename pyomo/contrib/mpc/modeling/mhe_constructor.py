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

from pyomo.core.base.set import Set
from pyomo.core.base.var import Var
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression

from pyomo.contrib.mpc.data.series_data import get_indexed_cuid


def curr_sample_point(tp, sample_points):
    """
    This function returns the sample point to which the given time element
    belongs.

    Arguments
    ---------
    tp: float
        Time element
    sample_points: iterable
        Set of sample points

    Returns
    -------
    A sample point to which the given time element belongs.

    """
    for spt in sample_points:
        if spt >= tp:
            return spt


def construct_measurement_variables_constraints(
    sample_points,
    variables,
):
    """
    This function constructs components for measurments, including:
    1. Index set for measurements
    2. Variables for measurements and measurement errors
    2. Constraints for measured variables, measurements, and measurement
    errors.

    Arguments
    ---------
    sample_points: iterable
        Set of sample points
    variables: list
        List of time-indexed variables that are measured

    Returns
    -------
    meas_set: Pyomo Set
        Index set for measurements
    meas_var: Pyomo Var
        Measurement variable, indexed by meas_set and sample_points
    meas_error_var: Pyomo Var
        Measurement error variable, indexed by meas_set and sample_points
    meas_con: Pyomo Constraint
        measurement constraint, indexed by meas_set and sample_points,
            **measurement == measured_var + measurement_error**

    """
    meas_set = Set(initialize=range(len(variables)))
    meas_var = Var(meas_set, sample_points)
    meas_error_var = Var(meas_set, sample_points, initialize=0.0)

    def meas_con_rule(mod, index, sp):
        return meas_var[index, sp] == \
            variables[index][sp] + meas_error_var[index, sp]
    meas_con = Constraint(
        meas_set, sample_points, rule=meas_con_rule,
    )

    return meas_set, meas_var, meas_error_var, meas_con


def construct_disturbed_model_constraints(
    time,
    sample_points,
    mod_constraints,
):
    """
    This function constructs components for model disturbances, including:
    1. Index set for model disturbances
    2. Variables for model disturbances
    3. Disturbed model constraints, consisting of original model constraints
    and model disturbances.

    Arguments
    ----------
    time: iterable
        Set by which to index model constraints
    sample_points: iterable
        Set of sample points
    mod_constraints : list
        List of model constraints to add model disturbances

    Returns
    -------
    disturbance_set: Pyomo Set
        Index set of model disturbances
    disturbance_var: Pyomo Var
        Model disturbance variable, indexed by disturbance_set and time
    disturbed_con: Pyomo Constraint
        Model constraints with model disturbances, indexed by disturbance_set
        and time,
            ** original_constraint + disturbance == 0.0 **

    """

    for con in mod_constraints:
        for t in time:
            con_ele = con[t]
            if not con_ele.equality:
                raise RuntimeError("Not an equality constraint: ", con_ele.name)

    disturbance_set = Set(initialize=range(len(mod_constraints)))
    disturbance_var = Var(disturbance_set, sample_points, initialize=0.0)

    def disturbed_con_rule(m, index, i):
        con_ele = mod_constraints[index][i]
        spt = curr_sample_point(i, sample_points)

        # Lower and upper bounds are the same if it's an equality
        return con_ele.body + disturbance_var[index, spt] == \
            con_ele.upper.value

    disturbed_con = Constraint(
        disturbance_set, time, rule=disturbed_con_rule,
    )

    return disturbance_set, disturbance_var, disturbed_con


def activate_disturbed_constraints_based_on_original_constraints(
    time,
    sample_points,
    disturbance_var,
    mod_constraints,
    disturbed_con,
):
    """
    This function activate the model constraint and deactivate the original
    constraint, if the original constarint is active. Also, if all model
    constraints within a specific sample period are not active, fix the
    disturbance variable at zero.

    Parameters
    ----------
    time: iterable
        Time set which indexes model constraints
    sample_points: iterable
        Set of sample points
    disturbance_var: Pyomo Var
        Model disturbances
    mod_constraints: list
        List of original model constraints
    disturbed_con: Pyomo Constraint
        Model constraints with model disturbances

    """
    # Deactivate original equalities and activate disturbed equalities
    for index, con in enumerate(mod_constraints):
        for tp in time:
            con_ele = con[tp]
            if con_ele.active:
                con_ele.deactivate()
                disturbed_con[index, tp].activate()
            else:
                # If the original equality is not active, deactivate the
                # disturbed one.
                disturbed_con[index, tp].deactivate()

    # If all constraints in a specific sample time are not active, fix that
    # model disturbance at zero.
    spt_saw_tp = {spt:[] for spt in sample_points}
    for tp in time:
        spt = curr_sample_point(tp, sample_points)
        spt_saw_tp[spt].append(tp)

    for index in range(len(mod_constraints)):
        for spt in sample_points:
            con_active_list = [
                disturbed_con[index, tp].active for tp in spt_saw_tp[spt]
            ]
            if not any(con_active_list):
                disturbance_var[index, spt].fix(0.0)


def get_cost_from_error_variables(variables, time, weight_data=None):
    """
    """
    # I know that these variables are indexed by time (aren't VarData), so I
    # don't need to send the time set to get_indexed_cuid
    cuids = [get_indexed_cuid(var) for var in variables]
    if weight_data is None:
        weight_data = {cuid: 1.0 for cuid in cuids}
    setpoint_data = {cuid: 0.0 for cuid in cuids}
    from pyomo.contrib.mpc.modeling.cost_expressions import (
        get_tracking_cost_from_constant_setpoint,
    )
    error_cost = get_tracking_cost_from_constant_setpoint(
        variables,
        time,
        setpoint_data,
        weight_data=weight_data,
    )
    return error_cost

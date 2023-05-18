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

from pyomo.contrib.mpc.interfaces.copy_values import (
    copy_values_at_time,
    _to_iterable,
    )
from pyomo.contrib.mpc.modeling.noise import (
    NoiseBoundOption,
    apply_noise_with_bounds,
    )
from pyomo.common.collections import ComponentMap


class DynamicVarLinker(object):
    """
    The purpose of this class is so that we do not have
    to call find_component or construct ComponentUIDs in a loop
    when transferring values between two different dynamic models.
    It also allows us to transfer values between variables that
    have different names in different models.

    """

    def __init__(self,
            source_variables,
            target_variables,
            source_time=None,
            target_time=None,
            ):
        # Right now all the transfers I can think of only happen
        # in one direction
        if len(source_variables) != len(target_variables):
            raise ValueError(
                "%s must be provided two lists of time-indexed variables "
                "of equal length. Got lengths %s and %s"
                % (type(self), len(source_variables), len(target_variables))
            )
        self._source_variables = source_variables
        self._target_variables = target_variables
        self._source_time = source_time
        self._target_time = target_time

    def _check_t_source_t_target(self, t_source, t_target):
        if t_source is None and self._source_time is None:
            raise RuntimeError(
                "Source time points were not provided in the transfer method "
                "or in the constructor."
            )
        elif t_source is None:
            t_source = self._source_time
        if t_target is None and self._target_time is None:
            raise RuntimeError(
                "Target time points were not provided in the transfer method "
                "or in the constructor."
            )
        elif t_target is None:
            t_target = self._target_time

        return t_source, t_target

    def transfer(self, t_source=None, t_target=None):
        t_source, t_target = self._check_t_source_t_target(t_source, t_target)

        copy_values_at_time(
            self._source_variables,
            self._target_variables,
            t_source,
            t_target,
        )

    def extract_data_from_source_variables_at_time(self, t_source):
        t_source = list(_to_iterable(t_source))
        data = ComponentMap(
            (var, [var[t].value for t in t_source])
            for var in self._source_variables
        )
        return data

    def apply_noise_to_extracted_data(self,
            data,
            noise_params,
            noise_function,
            bound_list,
            bound_option=NoiseBoundOption.DISCARD,
            max_number_discards=5,
            bound_push=0.0,
            ):

        noised_data = ComponentMap(
            (var, apply_noise_with_bounds(
                val_list=val_list,
                noise_params=[noise_params[idx]]*len(val_list),
                noise_function=noise_function,
                bound_list=[bound_list[idx]]*len(val_list),
                bound_option=bound_option,
                max_number_discards=max_number_discards,
                bound_push=bound_push,
                )
            )
            for idx, (var, val_list) in enumerate(data.items())
        )
        return noised_data

    def load_data_to_target_variables_at_time(self, data, t_target):
        t_target = list(_to_iterable(t_target))
        n_points = len(t_target)
        for svar, tvar in zip(self._source_variables, self._target_variables):
            val_list = data[svar]
            if len(val_list) == 1:
                val_list = val_list * n_points
            for t_t, val in zip(t_target, val_list):
                tvar[t_t].set_value(val)

    def transfer_with_noise(self,
            noise_params,
            noise_function,
            bound_list,
            t_source=None,
            t_target=None,
            bound_option=NoiseBoundOption.DISCARD,
            max_number_discards=5,
            bound_push=0.0,
            ):
        t_source, t_target = self._check_t_source_t_target(t_source, t_target)

        t_source = list(_to_iterable(t_source))
        t_target = list(_to_iterable(t_target))
        if (len(t_source) != len(t_target)
                and len(t_source) != 1):
            raise ValueError(
                "transfer_with_noise can only transfer data when lists of time\n"
                "points have the same length or the source list has length one."
            )

        data = self.extract_data_from_source_variables_at_time(t_source)
        noised_data = self.apply_noise_to_extracted_data(
            data,
            noise_params,
            noise_function,
            bound_list,
            bound_option,
            max_number_discards,
            bound_push,
        )
        self.load_data_to_target_variables_at_time(noised_data, t_target)

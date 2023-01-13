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
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.run_mpc import get_steady_state_data
from pyomo.contrib.mpc.examples.cstr.run_mhe import (
    get_control_inputs,
    run_cstr_mhe,
)


ipopt_available = pyo.SolverFactory("ipopt").available()


@unittest.skipIf(not ipopt_available, "ipopt is not available")
class TestCSTRMHE(unittest.TestCase):

    # This data was obtained from a run of this code. The test is
    # intended to make sure that values do not change, not that
    # they are correct in some absolute sense.
    _pred_sim_A_data = [
        1.15385, 1.25247, 1.31568, 1.35621, 1.38218, 1.39884, 1.49927,
        1.56205, 1.60128, 1.62580, 1.64112, 1.73239, 1.78804, 1.82198,
        1.84267, 1.85529, 1.93767, 1.98671, 2.01590, 2.03327, 2.04361,
        2.11838, 2.16185, 2.18712, 2.20181, 2.21036
    ]
    _pred_sim_B_data = [
        3.85615, 3.75753, 3.69432, 3.65379, 3.62782, 3.61116, 3.51072,
        3.44795, 3.40872, 3.38420, 3.36888, 3.27761, 3.22196, 3.18802,
        3.16733, 3.15471, 3.07233, 3.02329, 2.99410, 2.97673, 2.96639,
        2.89162, 2.84815, 2.82288, 2.80819, 2.79964
    ]
    _pred_estimate_A_data = [
        1.15385, 1.39884, 1.64112, 1.85529, 2.04361, 2.210357
    ]
    _pred_estimate_B_data = [
        3.85615, 3.61116, 3.36888, 3.15471, 2.96638, 2.799642
    ]

    def _get_initial_data(self):
        initial_data = mpc.ScalarData({"flow_in[*]": 0.3})
        return get_steady_state_data(initial_data)

    def test_mhe_simulation(self):
        initial_data = self._get_initial_data()
        sample_time = 2.0
        samples_per_horizon = 5
        ntfe_per_sample = 5
        ntfe_plant = 5
        simulation_steps = 5
        m_plant, sim_data, estimate_data = run_cstr_mhe(
            initial_data,
            samples_per_estimator_horizon=samples_per_horizon,
            sample_time=sample_time,
            ntfe_per_sample_estimator=ntfe_per_sample,
            ntfe_plant=ntfe_plant,
            simulation_steps=simulation_steps,
        )

        A_cuid = pyo.ComponentUID("conc[*,A]")
        B_cuid = pyo.ComponentUID("conc[*,B]")

        sim_time_points = [
            sample_time/ntfe_plant * i
            for i in range(simulation_steps*ntfe_plant + 1)
        ]

        sim_AB_data = sim_data.extract_variables([A_cuid, B_cuid])

        pred_sim_data = {
            A_cuid: self._pred_sim_A_data,
            B_cuid: self._pred_sim_B_data
        }

        self.assertStructuredAlmostEqual(
            pred_sim_data, sim_AB_data.get_data(), delta=1e-5
        )
        self.assertStructuredAlmostEqual(
            sim_time_points, sim_AB_data.get_time_points(), delta=1e-7
        )

        estimate_time_points = [
            sample_time * i
            for i in range(simulation_steps+1)
        ]

        estimate_AB_data = estimate_data.extract_variables([A_cuid, B_cuid])

        pred_estimate_data = {
            A_cuid: self._pred_estimate_A_data,
            B_cuid: self._pred_estimate_B_data
        }

        self.assertStructuredAlmostEqual(
            pred_estimate_data, estimate_AB_data.get_data(), delta=1e-5
        )
        self.assertStructuredAlmostEqual(
            estimate_time_points, estimate_AB_data.get_time_points(), delta=1e-7
        )


if __name__ == "__main__":
    unittest.main()


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
#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

import pyomo.common.unittest as unittest
import random

from pyomo.contrib.mpc.modeling.noise import (
    NoiseBoundOption,
    get_violated_bounds,
    MaxDiscardError,
    apply_noise,
    apply_noise_with_bounds,
    )


class TestNoise(unittest.TestCase):

    def test_get_violated_bounds(self):
        bounds = (1.0, 2.0)
        val = 1.5
        self.assertTupleEqual(get_violated_bounds(val, bounds), (None, 0))

        val = 0.8
        self.assertTupleEqual(get_violated_bounds(val, bounds), (1.0, 1))

        val = 2.5
        self.assertTupleEqual(get_violated_bounds(val, bounds), (2.0, -1))

    def test_apply_noise(self):
        random.seed(1234)
        noise_function = random.gauss
        val_list = [1.0, 2.0, 3.0]
        params_list = [0.05, 0.05, 0.05]
        new_vals = apply_noise(val_list, params_list, noise_function)

        for val, new_val, p in zip(val_list, new_vals, params_list):
            self.assertLessEqual(abs(val - new_val), 3 * p)

        noise_function = lambda val, rad: random.uniform(val - rad, val + rad)
        new_vals = apply_noise(val_list, params_list, noise_function)
        for val, new_val, p in zip(val_list, new_vals, params_list):
            self.assertLessEqual(abs(val - new_val), p)

    def test_apply_bounded_noise_discard(self):
        NBO = NoiseBoundOption
        random.seed(2345)
        noise_function = random.gauss
        vals = [5.0, 10.0]
        params = [1.0, 2.0]
        bounds_list = [(4.0, 6.0), (9.0, 11.0)]

        N = 100
        results = [set(), set()]
        for _ in range(N):
            # Very low probability we ever need to discard more than 15 times
            newvals = apply_noise_with_bounds(
                vals,
                params,
                noise_function,
                bounds_list,
                bound_option=NBO.DISCARD,
                max_number_discards=15,
            )
            for r, n in zip(results, newvals):
                r.add(n)

        for val, res, (lb, ub) in zip(vals, results, bounds_list):
            # Vanishingly small probability we get the same value twice.
            self.assertEqual(len(res), 100)

            inner_lb = val + (lb - val) / 2
            inner_ub = val + (ub - val) / 2
            n_inner = 0
            n_outer = 0
            for r in res:
                # Vanishingly small probability we land exactly on a bound
                self.assertLessEqual(lb, r)
                self.assertLessEqual(r, ub)
                if inner_lb < val and val < inner_ub:
                    n_inner += 1
                else:
                    n_outer += 1

            # Expect values to be clustered around the nominal.
            self.assertGreater(n_inner, n_outer)

        msg = "Max number of discards"
        with self.assertRaisesRegex(MaxDiscardError, msg):
            for _ in range(N):
                # Very likely that we will eventually need to discard a value
                newvals = apply_noise_with_bounds(
                    vals,
                    params,
                    noise_function,
                    bounds_list,
                    bound_option=NBO.DISCARD,
                    max_number_discards=0,
                )

        msg = "Max number of discards"
        with self.assertRaisesRegex(MaxDiscardError, msg):
            for _ in range(N):
                # In fact, very likely that we will need to discard more than
                # 5 times consecutively over the course of generating 100 numbers
                newvals = apply_noise_with_bounds(
                    vals,
                    params,
                    noise_function,
                    bounds_list,
                    bound_option=NBO.DISCARD,
                    max_number_discards=5,
                )

    def test_apply_bounded_noise_push(self):
        NBO = NoiseBoundOption
        random.seed(3456)
        noise_function = random.gauss
        vals = [5.0, 10.0]
        params = [1.0, 2.0]
        bounds_list = [(4.0, 6.0), (9.0, 11.0)]

        N = 100
        # Test with a zero bound push
        eps_b = 0.0
        results = [[], []]
        for _ in range(N):
            newvals = apply_noise_with_bounds(
                vals,
                params,
                noise_function,
                bounds_list,
                bound_option=NBO.PUSH,
                bound_push=eps_b,
            )
            for r, n in zip(results, newvals):
                r.append(n)

        # These flags will be used to make sure we cover both branches of an
        # "if tree" below.
        b1, b2 = False, False

        for val, res, p, (lb, ub) in zip(vals, results, params, bounds_list):
            # Very low probability we DON'T get the same value at least
            # twice. (I.e. we never exceed the same bound more than once.)
            self.assertLess(len(set(res)), 100)

            n_ub = 0
            n_lb = 0
            n_interior = 0
            for r in res:
                self.assertLessEqual(lb, r)
                self.assertLessEqual(r, ub)
                if r == lb:
                    n_lb += 1
                elif r == ub:
                    n_ub += 1
                else:
                    n_interior += 1

            self.assertGreaterEqual(n_lb, 1)
            self.assertGreaterEqual(n_ub, 1)
            self.assertGreaterEqual(n_interior, 1)

            # Very rough check that the distribution looks something like we
            # expect.
            if (val - lb) >= p and (ub - val) >= p:
                # If our bounds are at least sigma from the mean, we expect more
                # "interior" points than points at the bounds.
                self.assertGreater(n_interior, n_lb + n_ub)
                b1 = True
            elif (val - lb) <= p / 2 and (ub - val) <= p / 2:
                # If our bounds are within sigma/2 of the mean, we expect more
                # points at the bounds than in the interior.
                self.assertGreater(n_lb + n_ub, n_interior)
                b2 = True
            else:
                raise RuntimeError()

        # Sanity. Make sure we covered both branches.
        self.assertTrue(b1)
        self.assertTrue(b2)

        # Now test with a nonzero bound push
        eps_b = 0.01
        results = [[], []]
        for _ in range(N):
            newvals = apply_noise_with_bounds(
                vals,
                params,
                noise_function,
                bounds_list,
                bound_option=NBO.PUSH,
                bound_push=eps_b,
            )
            for r, n in zip(results, newvals):
                r.append(n)

        for val, res, p, (lb, ub) in zip(vals, results, params, bounds_list):
            # Very low probability we DON'T get the same value at least
            # twice. (I.e. we never exceed the same bound more than once.)
            self.assertLess(len(set(res)), 100)

            n_ub = 0
            n_lb = 0
            for r in res:
                # Satisfy bounds strictly
                self.assertLess(lb, r)
                self.assertLess(r, ub)
                if r == lb + eps_b:
                    n_lb += 1
                elif r == ub - eps_b:
                    n_ub += 1

            self.assertGreaterEqual(n_lb, 1)
            self.assertGreaterEqual(n_ub, 1)

    def test_apply_bounded_noise_fail(self):
        NBO = NoiseBoundOption
        random.seed(13456)
        noise_function = random.gauss
        vals = [5.0, 10.0]
        params = [1.0, 1.0]
        bounds_list = [(4.0, 6.0), (9.0, 11.0)]

        N = 10
        results = [[], []]
        msg = "Applying noise caused a bound to be violated"
        with self.assertRaisesRegex(RuntimeError, msg):
            # This is a very frail option
            for _ in range(N):
                newvals = apply_noise_with_bounds(
                    vals, params, noise_function, bounds_list, bound_option=NBO.FAIL
                )
                for r, n in zip(results, newvals):
                    r.append(n)

        N = 2
        # What if the RuntimeError raises within two iterations? KHL
        results = [[], []]
        for _ in range(N):
            newvals = apply_noise_with_bounds(
                vals, params, noise_function, bounds_list, bound_option=NBO.FAIL
            )
            for r, n in zip(results, newvals):
                r.append(n)

        for val, res, p, (lb, ub) in zip(vals, results, params, bounds_list):
            # Expect unique values that don't violate the bounds
            self.assertEqual(len(set(res)), N)
            for r in res:
                self.assertLess(lb, r)
                self.assertLess(r, ub)


if __name__ == "__main__":
    unittest.main()


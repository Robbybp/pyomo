#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.deprecation import deprecation_warning
deprecation_warning(
    'The pyomo.gdp.plugins.chull module is deprecated.  '
    'Import the Hull reformulation objects from pyomo.gdp.plugins.hull.',
    version='5.7')

from .hull import _Deprecated_Name_Hull as ConvexHull_Transformation

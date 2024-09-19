import numpy as np
import potpourri3d_bindings as pp3db

import scipy
import scipy.sparse

from .core import *

class DifferentialExteriorCalculus():

    def __init__(self, V, F):
        validate_mesh(V, F, force_triangular=True, test_indices=True)
        self.bound_solver = pp3db.DifferentialExteriorCalculus(V, F)

    def divergence(self, one_form):
        return self.bound_solver.divergence(one_form)
    
    def curl(self, one_form):
        return self.bound_solver.curl(one_form)

    def field_to_1form(self, field):
        return self.bound_solver.field_to_1form(field)

    def hodge_decomposition(self, one_form):
        return self.bound_solver.hodge_decomposition(one_form)

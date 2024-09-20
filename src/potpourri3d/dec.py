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

    def to_Edge1Form(self, field):
        return self.bound_solver.to_Edge1Form(field)
    
    def to_Face2Form(self, one_form):
        return self.bound_solver.to_Face2Form(one_form)

    def hodge_decomposition(self, one_form):
        return self.bound_solver.hodge_decomposition(one_form)
    
    def decompose_field(self, field):
        one_form = self.to_Edge1Form(field)
        exact, coexact, harmonic = self.hodge_decomposition(one_form)

        return (
            self.to_Face2Form(exact),
            self.to_Face2Form(coexact),
            self.to_Face2Form(harmonic)
        )

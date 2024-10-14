import numpy as np
import potpourri3d_bindings as pp3db

import scipy
import scipy.sparse

from .core import *

class DifferentialExteriorCalculus():

    def __init__(self, V, F):
        validate_mesh(V, F, force_triangular=True, test_indices=True)
        self.bound_solver = pp3db.DifferentialExteriorCalculus(V, F)

    @property
    def d0(self):
        return self.bound_solver.d0()
    
    @property
    def d1(self):
        return self.bound_solver.d1()
    
    @property
    def hodge0(self):
        return self.bound_solver.hodge0()
    
    @property
    def hodge1(self):
        return self.bound_solver.hodge1()
    
    @property
    def hodge2(self):
        return self.bound_solver.hodge2()
    
    @property
    def hodge0Inverse(self):
        return self.bound_solver.hodge0Inverse()
    
    @property
    def hodge1Inverse(self):
        return self.bound_solver.hodge1Inverse()
    
    @property
    def hodge2Inverse(self):
        return self.bound_solver.hodge2Inverse()
    
    @property
    def edge_faces(self):
        return self.bound_solver.edge_faces()
    
    @property
    def edge_vectors(self):
        return self.bound_solver.edge_vectors()

    def divergence(self, one_form):
        return self.bound_solver.divergence(one_form)
    
    def curl(self, one_form):
        return self.bound_solver.curl(one_form)

    def to_1Form(self, field):
        return self.bound_solver.to_1Form(field)
    
    def interpolate_1Form(self, one_form):
        return self.bound_solver.interpolate_1Form(one_form)

    def hodge_decomposition(self, one_form):
        return self.bound_solver.hodge_decomposition(one_form)
    
    def decompose_field(self, field):
        one_form = self.to_1Form(field)
        exact, coexact, harmonic = self.hodge_decomposition(one_form)

        return (
            self.interpolate_1Form(exact),
            self.interpolate_1Form(coexact),
            self.interpolate_1Form(harmonic)
        )

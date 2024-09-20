#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/trace_geodesic.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/eigen_interop_helpers.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace py = pybind11;

using namespace geometrycentral;
using namespace geometrycentral::surface;


// For overloaded functions, with C++11 compiler only
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;


// A wrapper class for the Differential Exterior Calculus
class DifferentialExteriorCalculus {

public:
  DifferentialExteriorCalculus(DenseMatrix<double> verts, DenseMatrix<int64_t> faces) {

    // Construct the internal mesh and geometry
    mesh.reset(new ManifoldSurfaceMesh(faces));
    geom.reset(new VertexPositionGeometry(*mesh));
    for (size_t i = 0; i < mesh->nVertices(); i++) {
      for (size_t j = 0; j < 3; j++) {
        geom->inputVertexPositions[i][j] = verts(i, j);
      }
    }
  }

  DenseMatrix<double> divergence(DenseMatrix<double> oneForm) {
    return geom->hodge0Inverse * geom->d0.transpose() * geom->hodge1 * oneForm;
  }

  DenseMatrix<double> curl(DenseMatrix<double> oneForm) {
    return geom->hodge2 * geom->d1 * oneForm;
  }

  // 2-form to 1-form
  Vector<double> toEdge1Form(DenseMatrix<double> field) {
    Vector<double> form = Vector<double>::Zero(mesh->nEdges());

    for (Edge e: mesh->edges()) {
      Halfedge h = e.halfedge();
      Vector3 f1, f2;

      if (h.isInterior()) {
        for (int i = 0; i < 3; i++) f1[i] = field(h.face().getIndex(), i);
      } else f1 = Vector3({0, 0, 0});

      if (h.twin().isInterior()) {
        for (int i = 0; i < 3; i++) f2[i] = field(h.face().getIndex(), i);
      } else f2 = Vector3({0, 0, 0});

      Vector3 vec = geom->halfedgeVector(h);

      form[e.getIndex()] = dot(f1 + f2, vec) * 0.5;
    }

    return form;
  }

  // 1-form to 2-form
  DenseMatrix<double> toFace2Form(DenseMatrix<double> oneForm) {
    DenseMatrix<double> field(mesh->nFaces(), 3);

    for (Face f: mesh->faces()) {
      Halfedge h = f.halfedge();

      Vector3 pi{geom->vertexPositions[h.vertex()]};
      Vector3 pj{geom->vertexPositions[h.next().vertex()]};
      Vector3 pk{geom->vertexPositions[h.next().next().vertex()]};
      Vector3 eij = pj - pi;
      Vector3 ejk = pk - pj;
      Vector3 eki = pi - pk;

      double cij = oneForm(h.edge().getIndex(), 0);
      double cjk = oneForm(h.next().edge().getIndex(), 0);
      double cki = oneForm(h.next().next().edge().getIndex(), 0);
      if (h.edge().halfedge() != h) cij *= -1;
      if (h.next().edge().halfedge() != h.next()) cjk *= -1;
			if (h.next().next().edge().halfedge() != h.next().next()) cki *= -1;

      Vector3 a = (eki - ejk) * cij;
      Vector3 b = (eij - eki) * cjk;
      Vector3 c = (ejk - eij) * cki;

      double A = geom->faceArea(f);
      Vector3 N = geom->faceNormal(f);
      Vector3 vec = cross(N, a + b + c) / (6 * A);

      field(f.getIndex(), 0) = vec[0];
      field(f.getIndex(), 1) = vec[1];
      field(f.getIndex(), 2) = vec[2];
    }

    return field;
  }

  // Perform Hodge decomposition
  std::tuple<Vector<double>, Vector<double>, Vector<double>> hodgeDecomposition(Vector<double> oneForm) {
    geom->requireDECOperators();

    //// EXACT
    // Define equation: Ax = a
    SparseMatrix<double> A = (geom->d0.transpose() * geom->hodge1 * geom->d0);
    SparseMatrix<double> identityMatrix(A.rows(), A.cols());
    identityMatrix.setIdentity();
    A = A + (identityMatrix * 1e-8);

    Vector<double> a = geom->d0.transpose() * geom->hodge1 * oneForm;

    // Solve
    PositiveDefiniteSolver<double> choleskySolver(A);
    Vector<double> exact = geom->d0 * choleskySolver.solve(a);

    //// COEXACT
    // Define equation: Bx = b
    SparseMatrix<double> B = geom->d1 * geom->hodge1Inverse * geom->d1.transpose();
    Vector<double> b = geom->d1 * oneForm;

    // Solve
    SquareSolver<double> LUSolver(B);
    Vector<double> coexact = geom->hodge1Inverse * geom->d1.transpose() * LUSolver.solve(b);

    //// Harmonic
    Vector<double> harmonic = oneForm - (exact + coexact); 

    return std::tuple<Vector<double>, Vector<double>, Vector<double>> (exact, coexact, harmonic);
  }

private:
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
};

// Actual binding code
// clang-format off
void bind_dec(py::module& m) {
  py::class_<DifferentialExteriorCalculus>(m, "DifferentialExteriorCalculus")
        .def(py::init<DenseMatrix<double>, DenseMatrix<int64_t>>())
        .def("divergence", &DifferentialExteriorCalculus::divergence, py::arg("oneForm"))
        .def("curl", &DifferentialExteriorCalculus::curl, py::arg("oneForm"))
        .def("to_Edge1Form", &DifferentialExteriorCalculus::toEdge1Form, py::arg("field"))
        .def("to_Face2Form", &DifferentialExteriorCalculus::toFace2Form, py::arg("one_form"))
        .def("hodge_decomposition", &DifferentialExteriorCalculus::hodgeDecomposition, py::arg("one_form"));
}
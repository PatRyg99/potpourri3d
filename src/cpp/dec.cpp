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
    mesh.reset(new SurfaceMesh(faces));
    geom.reset(new VertexPositionGeometry(*mesh));
    for (size_t i = 0; i < mesh->nVertices(); i++) {
      for (size_t j = 0; j < 3; j++) {
        geom->inputVertexPositions[i][j] = verts(i, j);
      }
    }

    geom->requireFaceIndices();  
    geom->requireEdgeIndices();   
  }

  DenseMatrix<double> divergence(DenseMatrix<double> oneForm) {
    return geom->hodge0Inverse * geom->d0.transpose() * geom->hodge1 * oneForm;
  }

  DenseMatrix<double> curl(DenseMatrix<double> oneForm) {
    return geom->hodge2 * geom->d1 * oneForm;
  }

  // 2-form to 1-form
  DenseMatrix<double> fieldTo1Form(DenseMatrix<double> field) {
    DenseMatrix<double> E(mesh->nEdges(), 1);
    FaceData<size_t> faceIndex = geom->faceIndices;
    EdgeData<size_t> edgeIndex = geom->edgeIndices;

    for (Edge e: mesh->edges()) {
      Halfedge h = e.halfedge();

      Vector3 f1{
        field(faceIndex[h.face()], 0),
        field(faceIndex[h.face()], 1),
        field(faceIndex[h.face()], 2)
      };  
      Vector3 f2{
        field(faceIndex[h.twin().face()], 0),
        field(faceIndex[h.twin().face()], 1),
        field(faceIndex[h.twin().face()], 2)
      };
      Vector3 direction = {
        geom->vertexPositions[h.next().vertex()] - geom->vertexPositions[h.vertex()]
      };

      E(edgeIndex[e], 0) = dot((f1 + f2), direction) / 2;
    }

    return E;
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
        .def("field_to_1form", &DifferentialExteriorCalculus::fieldTo1Form, py::arg("field"))
        .def("hodge_decomposition", &DifferentialExteriorCalculus::hodgeDecomposition, py::arg("one_form"));
}
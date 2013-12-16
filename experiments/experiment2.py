from dolfin import *
import numpy
from matplotlib import pyplot
import cPickle
from collections import defaultdict
import unittest

import robert_visualize_transformation

# This dependence will be removed once we implement the proper
# buffering of the boundary condition polygons with CGAL
from shapely import geometry

# Optimization options for the form compiler

parameters["form_compiler"]["cpp_optimize"] = True

ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# read in the boundary conditions

with open("table2.data", 'r') as f:
    obj = cPickle.load(f)

b = obj['bounds']
x_min, y_min, x_max, y_max = b[0], b[1], b[2], b[3]

obj.pop('bounds', None)

boundary = defaultdict()

range_boundary = defaultdict()

# This code will be removed, one the CGAL extension is in place:
boundary_buff = defaultdict()

for (key, val) in obj.iteritems():
    # only the untransformed region for now
    points = map(lambda x: Point(*x), val[0][0:-1])
    boundary[key] = Polygon(points)

for (key, val) in obj.iteritems():
    # the tranformed region
    points = map(lambda x: Point(*x), val[1][0:-1])
    range_boundary[key] = Polygon(points)

# This code will be removed once the buffering is done with CGAL:
for (key, val) in obj.iteritems():
    polygon = geometry.Polygon(val[0][0:-1])
    boundary_buff[key] = polygon.buffer(0.1)

# This code will be removed
domain = geometry.box(x_min, y_min, x_max, y_max)
for (key, val) in obj.iteritems():
    polygon = geometry.Polygon(val[0][0:-1])
    domain = domain.difference(polygon)

outer_frame = Rectangle(x_min, y_min, x_max, y_max)

# calculate the domain of the initial value problem
Omega = outer_frame
for (key, val) in boundary.iteritems():
    Omega = Omega - boundary[key]

# Calculate the region of the range
Gamma = outer_frame
for (key, val) in range_boundary.iteritems():
    Gamma = Gamma - range_boundary[key]

# generate the mesh
mesh = Mesh(Omega, 50)
range_mesh = Mesh(Gamma, 50)

# plot the domain
# plot(mesh, title="2D mesh", interactive=True)
# plot(range_mesh, title="2D mesh", interactive=True)

V = VectorFunctionSpace(mesh, "Lagrange", 1)

outer_boundary = CompiledSubDomain("(((x[0] <= x_min + DOLFIN_EPS) || (x[0] >= x_max - DOLFIN_EPS) || (x[1] <= y_min + DOLFIN_EPS) || (x[1] >= y_max - DOLFIN_EPS))) && on_boundary", x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max)

# this is likely to be slow. Later, it should be implemented in C
# code, using CGAL's create_exterior_skeleton_and_offset_polygons_2.

def inner_boundary(x, key):
    return boundary_buff[key].contains(geometry.Point(x))

# same for this, which should also be done directly in fenics/CGAL
def evaluate_on_boundary(domain_poly, range_poly, x):
    gamma = geometry.LineString(domain_poly).project(geometry.Point(x))
    acc = 0.0
    for i in range(len(domain_poly) - 1):
        v = numpy.array(domain_poly[i])
        w = numpy.array(domain_poly[i+1])
        dist = numpy.linalg.norm(v - w)
        if acc + dist >= gamma - DOLFIN_EPS:
            nu = (gamma - acc)/dist;
            result = numpy.array(range_poly[i]) + \
                nu * (numpy.array(range_poly[i+1]) - numpy.array(range_poly[i]))
            # print result
            return result
        acc = acc + dist

class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.domain_poly = \
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0),
             (0.0, 1.0), (0.0, 0.0)]
        self.range_poly = \
            [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0),
             (1.0, 2.0), (1.0, 1.0)]
    def test_evaluate_on_boundary(self):
        point = numpy.array([0.0, 0.0])
        result = evaluate_on_boundary(self.domain_poly, self.range_poly, point)
        self.assertEqual(result, (1.0, 1.0))

# This is also slow (see fenics documentation)
class PolygonExpression(Expression):
    def __init__(self, domain, ran):
        self._domain = domain
        self._range = ran
    def eval(self, value, x):
        v = evaluate_on_boundary(self._domain, self._range, x)
        value[0] = v[0]
        value[1] = v[1]
    # This is a vector:
    def value_shape(self):
        return (2,)
        

u_outer = Expression(("x[0]", "x[1]"))
u0 = PolygonExpression(domain=obj['rect'][0], ran=obj['rect'][1])
u1 = PolygonExpression(domain=obj['rope'][0], ran=obj['rope'][1])
u2 = PolygonExpression(domain=obj['circ'][0], ran=obj['circ'][1])

bc_outer = DirichletBC(V, u_outer, outer_boundary)
bc0 = DirichletBC(V, u0, lambda x: inner_boundary(x, 'rect'))
bc1 = DirichletBC(V, u1, lambda x: inner_boundary(x, 'rope'))
bc2 = DirichletBC(V, u2, lambda x: inner_boundary(x, 'circ'))

   
du = TrialFunction(V)
v = TestFunction(V)
u = Function(V)


a, b = 100, -1

A  = Constant((1.0, 0.0))
B  = Constant((0.0, 1.0))

# Define normal component, mesh size
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
n = FacetNormal(mesh)
range_n = FacetNormal(range_mesh)

# Penalty parameter
alpha = Constant(8.0)

# Kinematics
I = Identity(V.cell().d) # identity tensor
F = I + grad(u) # Deformation gradient
C = F.T * F # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

psi = a * (Ic - e) - 1000 * ln(J)
Pi1 = 1e-10 * psi * dx \
    + inner(div(grad(dot(A, u))), div(grad(dot(A, v)))) * dx \
    +  inner(div(grad(dot(B, u))), div(grad(dot(B, v)))) * dx \
    - inner(jump(grad(dot(A, u)), n), avg(div(grad(dot(A, v)))))*dS \
    - inner(jump(grad(dot(B, u)), n), avg(div(grad(dot(B, v)))))*dS \
    - inner(avg(div(grad(dot(A, u)))), jump(grad(dot(A, v)), n))*dS \
    - inner(avg(div(grad(dot(B, u)))), jump(grad(dot(B, v)), n))*dS # \
#    + alpha/h_avg('+')*inner(jump(grad(dot(A, u)),n), jump(grad(dot(A, v)),n))*dS \
#    + alpha/h_avg('+')*inner(jump(grad(dot(B, u)),n), jump(grad(dot(B, v)),n))*dS

# psi * dx - dot(B, u) * dx - dot(T, u)*ds

# Kinematics
I = Identity(V.cell().d) # identity tensor
F = grad(u) + I # Deformation gradient
C = F.T * F # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

B  = Constant((1.0, 1.0))
T  = Constant((1.0,  1.0))


psi = - 100 * ln(J) + a * Ic #  - J
psi = a * Ic - 100 * ln(J)
Pi2 = psi * dx # - 10 * dot(B, u) * dx - dot(T, u)*ds

psi = a * Ic - 100 * ln(J)
Pi3 = psi * dx + 100000 * inner(dot(grad(u), n) - range_n, dot(grad(u), n) - range_n) * ds

def calc_solution(Pi):
    u = Function(V)
    # Compute first variation of Pi
    F = derivative(Pi, u, v)

    # Compute Jacobian of F
    J = derivative(F, u, du)

    # Solve variational problem
    #problem = NonlinearVariationalProblem(F, u, [bc_outer, bc0, bc1, bc2], J)
    # solver  = NonlinearVariationalSolver(problem)
    #solver.solve()
    solve(F == 0, u, [bc_outer, bc0, bc1], J=J,
          form_compiler_parameters=ffc_options)
    return u

# u1 = calc_solution(Pi1)
# print u1
# print u1(0.5, 0.5)
# u2 = calc_solution(Pi2)
# print u2

def calc_f(u, x):
    X = 10 * x[0]
    Y = 10 * x[1]
    if domain.contains(geometry.Point(numpy.array([X, Y]))):
        return u(X, Y)/10
    else:
        return numpy.array([0.0, 0.0])

def g(x):
    X = 10 * x[0]
    Y = 10 * x[1]
    return not(domain.contains(geometry.Point(numpy.array([X, Y]))))

fig = pyplot.figure()
ax1 = fig.add_subplot(1,3,1)
ax1.set_aspect('equal')
ax2 = fig.add_subplot(1,3,2)
ax2.set_aspect('equal')
ax3 = fig.add_subplot(1,3,3)
ax3.set_aspect('equal')

# robert_visualize_transformation.visualize_transformation(ax1, \
#     lambda x: calc_f(u1, x), g)

# u = Function(V)
# u2 = calc_solution(Pi2)
# print u2
# print u2(0.5, 0.5)

# print calc_f(u2, numpy.array([0.5, 0.5]))

# robert_visualize_transformation.visualize_transformation(ax2, \
#    lambda x: calc_f(u2, x), g)

u3 = calc_solution(Pi3)

robert_visualize_transformation.visualize_transformation(ax3, \
    lambda x: calc_f(u3, x), g)

pyplot.show()

# plot(u, interactive=True)


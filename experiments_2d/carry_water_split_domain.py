import robert_visualize_transformation
from matplotlib import pyplot

from dolfin import *

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

mesh = UnitSquareMesh(25, 25)

V = VectorFunctionSpace(mesh, "Lagrange", 1)

parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
parts.set_all(0)

class outer_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > DOLFIN_EPS and on_boundary

class bottom_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < DOLFIN_EPS and abs(x[0] - 0.5) > 0.25 and on_boundary

class bottom_middle_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < DOLFIN_EPS and abs(x[0] - 0.5) < 0.25 and on_boundary

outer_b = outer_boundary()
bottom_b = bottom_boundary()
bottom_middle_b = bottom_middle_boundary()
outer_b.mark(parts, 1)
bottom_b.mark(parts, 2)
bottom_middle_b.mark(parts, 3)

#plot(parts, interactive=True)

outer = Expression(("x[0]", "x[1]"))
bc_outer = DirichletBC(V, outer, parts, 1)
bottom = Expression(("x[0]", "x[1]"))
bc_bottom = DirichletBC(V, bottom, parts, 2)
bottom_middle = Expression(("x[0]", "0.1 + 0.1*cos(4*pi*(x[0] - 0.25) - pi)"))
bc_bottom_middle = DirichletBC(V, bottom_middle, parts, 3)

bc = [bc_outer, bc_bottom, bc_bottom_middle]

class Omega1(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= 0.5
class Omega2(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= 0.5

subdomains = MeshFunction('size_t', mesh, mesh.topology().dim())
#subdomains.init(mesh.topology().dim())
subdomains.set_all(0)

subdomain1 = Omega1()
subdomain1.mark(subdomains, 1)
subdomain2 = Omega2()
subdomain2.mark(subdomains, 2)

dx_new = Measure('dx')[subdomains]

du = TrialFunction(V)
v  = TestFunction(V)
u  = Function(V)

F = grad(u)
C = F.T*F

Ic = tr(C)

zvec = Constant((0.0, 1.0))
xvec = Constant((1.0, 0.0))
weight1 = Expression("(x[0]<=0.5)")
weight2 = Expression("(x[0]>=0.5)")

psi8a = Ic + (det(grad(u)) - 1)*(det(grad(u)) - 1)
psi8b = Ic + dot(grad(u)*zvec, xvec)**2
#Pi8 = psi8a*dx(1) + psi8b*dx(2)
#Pi8 = psi8a*dx_new(1) + psi8b*dx_new(2)
#Pi8 = Ic*dx_new(1) + Ic*dx_new(2)
#Pi8 = weight1*psi8a*dx + weight2*psi8b*dx
Pi8 = weight1*psi8a*dx + weight2*psi8b*dx

Pi = Pi8

F = derivative(Pi, u, v)
J = derivative(F, u, du)

solve(F == 0, u, bc, J=J)

# plot(u, mode = "displacement", interactive=True)

fig = pyplot.figure()
ax1 = fig.add_subplot(1,3,1)
ax1.set_aspect('equal')

def try_evaluate(point):
    try:
        u(point[0], point[1])
    except:
        return True
    return False

robert_visualize_transformation.visualize_transformation(ax1, u, try_evaluate)
pyplot.show()

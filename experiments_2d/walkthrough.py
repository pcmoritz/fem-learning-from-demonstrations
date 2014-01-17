def main():
    import meshpy.triangle as triangle

    points = [(0,0), (1,0), (1,1), (0,1)]

    left_points = [(0.2, 0.25), (0.4, 0.25), (0.4, 0.5), (0.2, 0.5)]
    right_points = [(0.8, 0.25), (0.6, 0.25), (0.6, 0.5), (0.8, 0.5)]

    def round_trip_connect(start, end):
      result = []
      for i in range(start, end):
        result.append((i, i+1))
      result.append((end, start))
      return result

    info = triangle.MeshInfo()
    info.set_points(points + left_points + right_points)
    outer_facets = round_trip_connect(0, len(points)-1)
    left_facets = round_trip_connect(len(points), len(points) + len(left_points) - 1)
    right_facets = round_trip_connect(len(points) + len(left_points), len(points) + len(left_points) + len(right_points) - 1)
    info.set_facets(outer_facets + left_facets + right_facets)
    info.set_holes([(0.3, 0.3)] + [(0.7, 0.3)])

    mesh = triangle.build(info, max_volume=1e-3, min_angle=25)

    f = open('output.xml', 'w')

    f.write("""
        <?xml version="1.0" encoding="UTF-8"?>

        <dolfin xmlns:dolfin="http://www.fenics.org/dolfin/">
          <mesh celltype="triangle" dim="2">
            <vertices size="%d">
        """ % len(mesh.points))

    for i, pt in enumerate(mesh.points):
      f.write('<vertex index="%d" x="%g" y="%g"/>' % (
              i, pt[0], pt[1]))

    f.write("""
        </vertices>
        <cells size="%d">
        """ % len(mesh.elements))

    for i, element in enumerate(mesh.elements):
      f.write('<triangle index="%d" v0="%d" v1="%d" v2="%d"/>' % (
              i, element[0], element[1], element[2]))

    f.write("""
            </cells>
          </mesh>
        </dolfin>
        """)

from dolfin import *

main()

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

mesh = Mesh("output.xml")

V = VectorFunctionSpace(mesh, "Lagrange", 1)

parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
parts.set_all(0)

class outer_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - 0.5) >= 0.4 or abs(x[1] - 0.5) >= 0.4) and on_boundary

class left_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - 0.3) <= 0.15 and abs(x[1] - 0.375) <= 0.13) and on_boundary

class right_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - 0.7) <= 0.15 and abs(x[1] - 0.375) <= 0.13) and on_boundary

outer_b = outer_boundary()
left_b = left_boundary()
right_b = right_boundary()
outer_b.mark(parts, 1)
left_b.mark(parts, 2)
right_b.mark(parts, 3)

# plot(parts, interactive=True)

outer = Expression(("x[0]", "x[1]"))
bc_outer = DirichletBC(V, outer, parts, 1)
left = Expression(("x[0]+0.05", "x[1]"))
bc_left = DirichletBC(V, left, parts, 2)
right = Expression(("x[0]-0.05", "x[1]"))
bc_right = DirichletBC(V, right, parts, 3)

bc = [bc_outer, bc_left, bc_right]

du = TrialFunction(V)
v  = TestFunction(V)
u  = Function(V)

I = Identity(V.cell().d)
F = I + grad(u)
C = F.T*F

Ic = tr(C)
J  = det(F)

psi = Ic - 2
Pi = psi*dx + 8*(det(grad(u)) - 1)*(det(grad(u)) - 1)*dx

F = derivative(Pi, u, v)
J = derivative(F, u, du)

solve(F == 0, u, bc, J=J)

# plot(u, mode = "displacement", interactive=True)

import robert_visualize_transformation
from matplotlib import pyplot

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

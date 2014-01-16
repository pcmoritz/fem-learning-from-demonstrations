from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = UnitSquareMesh(16, 16)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition
u0 = Function(V)
# Mark boundary subdomians
left_right_top, bottom = compile_subdomains(["(x[0] < DOLFIN_EPS) || (x[0] > 1 - DOLFIN_EPS) || (x[1] > 1 - DOLFIN_EPS)", "(x[0] < DOLFIN_EPS)"])

left_right_top_condition = Expression(("0.0", "0.0"))
#left_right_top_condition = Expression(("x[0]", "x[1]"))
bottom_condition = Expression(("0.0", "scale*sin(pi*x[0])"), scale = 0.2)
#bottom_condition = Expression(("x[0]", "scale*sin(pi*x[0])"), scale = 0.2)

bc_left_right_top = DirichletBC(V, left_right_top_condition, left_right_top)
bc_bottom = DirichletBC(V, bottom_condition, bottom)

bcs = [bc_left_right_top, bc_bottom]

# Define variational problem
u_displacement = TrialFunction(V)
v_test = TestFunction(V)

# Define function for the solution
u_solution = Function(V)

# Kinematics
I = Identity(V.cell().d)
F = I + grad(u_solution)
C = F.T*F

# Invariants of deformation tensors
Ic = tr(C)
J = det(F)

# Simple objective
psi = Ic
Pi = psi*dx

# Solve
F = derivative(Pi, u_solution, v_test)
J = derivative(F, u_solution, u_displacement)
solve(F == 0, u_solution, bcs, J=J)

# Plot
plot(u_solution, mode = "displacement", interactive = True)


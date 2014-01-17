from dolfin import *

# Optimization options for the form compiler
#parameters["form_compiler"]["cpp_optimize"] = True
#ffc_options = {"optimize": True, \
#               "eliminate_zeros": True, \
#               "precompute_basis_const": True, \
#               "precompute_ip_const": True}

# Create mesh and define function space
mesh = UnitSquareMesh(16, 16)
V = VectorFunctionSpace(mesh, "Lagrange", 2)

boundary_parts = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
boundary_parts.set_all(2)

def BCbottom(x, on_boundary):
    return x[0] < DOLFIN_EPS and on_boundary

def BCtop(x, on_boundary):
    return x[0] > 1.0 - DOLFIN_EPS and on_boundary

def BCleft(x, on_boundary):
    return x[1] < DOLFIN_EPS and on_boundary

def BCright(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS and on_boundary

# Define boundary condition
u0 = Function(V)
# Mark boundary subdomians
#left_right_top, bottom = compile_subdomains(["((x[1] < DOLFIN_EPS) || (x[1] > 1 - DOLFIN_EPS) || (x[0] > 1 - DOLFIN_EPS)) && on_boundary", "(x[0] < DOLFIN_EPS) && on_boundary"])
bndry = compile_subdomains(["on_boundary"])

#left_right_top_condition = Expression(("0.0", "0.0"))
#left_right_top_condition = Expression(("x[0]", "x[1]"))
#bottom_condition = Expression(("0.0", "0.0"))
#bottom_condition = Expression(("x[0]", "x[1]"))
#bottom_condition = Expression(("0.0", "0.01*sin(pi*x[0])"))
#bottom_condition = Expression(("x[0]", "scale*sin(pi*x[0])"), scale = 0.2)
#bndry_condition = Expression(("0.0", "0.0"))
bndry_condition = Expression(("x[0]", "x[1]"))

#bc_left_right_top = DirichletBC(V, left_right_top_condition, left_right_top)
#bc_bottom = DirichletBC(V, bottom_condition, bottom)
bc_bndry = DirichletBC(V, bndry_condition, bndry)

#bcs = [bc_left_right_top, bc_bottom]
bcs = [bc_bndry]

# Define variational problem
u_displacement = TrialFunction(V)
v_test = TestFunction(V)

# Define function for the solution
u_solution = Function(V)

# Kinematics
#I = Identity(V.cell().d)
#F = I + grad(u_solution)
#C = F.T*F

# Elasticity parameters
#E, nu = 10.0, 0.3
#mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Invariants of deformation tensors
#Ic = tr(C)
#J = det(F)

# Simple objective
#psi = (mu/2)*(Ic - 2)# - mu*ln(J) + (lmbda/2)*(ln(J))**2
#Pi = psi*dx
#Pi = Expression(("pow(u_solution[0],2) + pow(u_solution[1],2)"))*dx
tempv = Constant((0.5,0.5))
psi = dot(u_solution, u_solution)
#Pi = Constant(0.0)*dx
#Pi = tr(10*I + grad(u_solution).T * grad(u_solution))*dx
Pi = psi*dx

# Solve
F = derivative(Pi, u_solution, v_test)
J = derivative(F, u_solution, u_displacement)
solve(F == 0, u_solution, bcs, J=J)

# Plot
#plot(u_solution, mode = "displacement", interactive = True)
plot(u_solution, interactive = True)


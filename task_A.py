from dune.grid import cartesianDomain
from dune.fem.function import integrate
from dune.ufl import Constant
from ufl import as_vector, sqrt, dot

from chorins_method import setup

# Define domain
H = 1; L = 1
domain = cartesianDomain([0, 0], [L, H], [16, 16])

# Define parameters
μ = Constant(1, name="mu")
ρ = Constant(1, name="rho")
f = Constant([0, 0], name="source")

dt = Constant(0.02, name="dt")
t = Constant(0, name="time")
T = 10

problem = setup(domain)
view, x = next(problem)

left = x[0] <= 0
right = x[0] >= L
top = x[1] >= H
bottom = x[1] <= 0

bc_velocity = {top: [0, 0], bottom: [0, 0]}
bc_pressure = {left: 8, right: 0}

u, p, step = problem.send((bc_velocity, bc_pressure, μ, ρ, t, dt, f))

u.interpolate([0, 0])
p.interpolate(0)

u_true = as_vector([4 * x[1] * (1 - x[1]), 0])
p_true = 8 * (1 - x[0])
error_velocity = lambda u: integrate(view, sqrt(dot(u - u_true, (u - u_true))), order=3)

for i in range(round(T/dt)):
    step()
    if i % 100 == 0:
        print("Finished step: ", i, " Error: ", error_velocity(u))

u.plot()
p.plot()

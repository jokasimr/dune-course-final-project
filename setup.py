import resource
previous_memory_consumption = None
def print_memory_usage(when):
    global previous_memory_consumption
    memory = comm.max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    diff = memory - previous_memory_consumption if previous_memory_consumption else None
    if comm.rank == 0:
        print("peak memory usage: ", when, ": ", memory, " kb", f"(diff {diff} kb)")
    previous_memory_consumption = memory
    
import time
from itertools import islice
import numpy as np
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid, aluCubeGrid
from dune.fem.view import adaptiveLeafGridView
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.fem.function import integrate, uflFunction
from dune import fem
from dune.common import comm

from dune.ufl import DirichletBC, Constant
from ufl import TestFunction, TrialFunction, SpatialCoordinate,\
      FacetNormal, dx, ds, div, nabla_grad, grad, dot, inner, sqrt, exp, sin, pi, as_vector, \
      conditional, curl, Identity, CellVolume


def setup(domain, grid_type=aluConformGrid):
    if comm.rank == 0:
        view = adaptiveLeafGridView(grid_type(domain))
    else:
        view = adaptiveLeafGridView(grid_type({"vertices": [], "simplices": []}, dimgrid=2))

    fem.loadBalance(view.hierarchicalGrid)
    dim = view.dimension

    # Spaces
    storage = "petsc"
    space_velocity = lagrange(view, order=2, dimRange=dim, storage=storage)
    x = SpatialCoordinate(space_velocity)
    n = FacetNormal(space_velocity)
    u = TrialFunction(space_velocity)
    v = TestFunction(space_velocity)

    space_pressure = lagrange(view, order=1,  storage=storage)
    p = TrialFunction(space_pressure)
    q = TestFunction(space_pressure)
    
    bc_velocity, bc_pressure, μ, ρ, t, dt, f = yield (view, x)
    
    # Variational problems
    def tentative_velocity_form(u_old, p_old):
        ϵ = lambda u: (grad(u) + grad(u).T) / 2
        σ = lambda u, p: 2 * μ * ϵ(u) - p * Identity(u_old.dimRange)
        u_avg = (u + u_old) / 2
        
        return (
            inner(ρ * (u - u_old) / dt, v) * dx
            + inner(ρ * grad(u_old) * u_old, v) * dx
            + inner(σ(u_avg, p_old), ϵ(v)) * dx
            + inner(p_old * n - μ * nabla_grad(u_avg) * n, v) * ds
            - inner(f, v) * dx
        ) == 0

    def pressure_update_form(p_old, u_tentative):
        return (
            inner(grad(p) - grad(p_old), grad(q))
            + 1/dt * inner(div(u_tentative), q)
        ) * dx == 0

    def velocity_update_form(u_tentative, p_new, p_old):
        return (
            inner(u - u_tentative, v)
            + dt * inner(grad(p_new - p_old), v)
        ) * dx == 0

    # Vectors
    u_old = space_velocity.interpolate([0]*dim, name="old_velocity")
    u_tentative = space_velocity.interpolate([0]*dim, name="tentative_velocity")
    u_new = space_velocity.interpolate([0]*dim, name="new_velocity")
    p_old = space_pressure.interpolate(0, name="old_pressure")
    p_new = space_pressure.interpolate(0, name="new_pressure")
    
    # Boundary conditions
    u_bc = [
        DirichletBC(space_velocity, bc, where)
        for where, bc in bc_velocity.items()
    ]
    p_bc = [
        DirichletBC(space_pressure, bc, where)
        for where, bc in bc_pressure.items()
    ]

    # Solvers
    shared_solver_parameters = {
        "newton.tolerance": 1e-5,
        "newton.linear.tolerance": 1e-8,
        "newton.verbose": True,
        #"newton.linear.preconditioning.method": "ssor",
        #"newton.linear.verbose": True,
        #"newton.linear.preconditioning.method": "ssor",
        #"newton.linear.preconditioning.hypre.method": "boomeramg",
    }
    
    tentative_velocity_problem = galerkin(
        [tentative_velocity_form(u_old, p_old), *u_bc],
        solver="gmres",
        parameters={
            # slower than without
            #"newton.linear.preconditioning.method": "jacobi",
            #"newton.linear.preconditioning.method": "amg-ilu",
            **shared_solver_parameters
        }
    )
    pressure_problem = galerkin(
        [pressure_update_form(p_old, u_tentative), *p_bc],
        solver="cg",
        parameters={
            # slower than without
            #"newton.linear.preconditioning.method": "jacobi",
            **shared_solver_parameters
        }
    )
    velocity_problem = galerkin(
        [velocity_update_form(u_tentative, p_new, p_old), *u_bc],
        solver="gmres",
        parameters={
            # slower than without
            #"newton.linear.preconditioning.method": "sor",
            **shared_solver_parameters
        }
    ) 
    
    def step():
        print_memory_usage("before step")
        if comm.rank == 0:
            print("tentative solve")
        start = time.time()
        tentative_velocity_problem.solve(target=u_tentative)
        print_memory_usage("after tentative")
        maxtime = comm.max(time.time() - start)
        if comm.rank == 0:
            print("tentative solve", maxtime)
            print("pressure solve")
        start = time.time()
        pressure_problem.solve(target=p_new)
        print_memory_usage("after pressure")
        maxtime = comm.max(time.time() - start)
        if comm.rank == 0:
            print("pressure solve", maxtime)
            print("velocity solve")
        start = time.time()
        velocity_problem.solve(target=u_new)
        print_memory_usage("after velocity")
        maxtime = comm.max(time.time() - start)
        if comm.rank == 0:
            print("velocity solve", maxtime)
        u_old.assign(u_new)
        p_old.assign(p_new)
        print_memory_usage("after assignment")
        t.value += dt.value
        
    yield u_old, p_old, u_tentative, u_new, p_new, step

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
    phi = p
    
    bc_velocity, bc_pressure, mu, rho, t, dt, f = yield (view, x)
    
    # See https://www.dealii.org/current/doxygen/deal.II/step_35.html
    # Vectors
    ukm1 = space_velocity.interpolate([0]*dim, name="u_km1")
    uk = space_velocity.interpolate([0]*dim, name="u_k")
    ukp1 = space_velocity.interpolate([0]*dim, name="u_kp1")
    
    pk = space_pressure.interpolate(0, name="p_k")
    pkp1 = space_pressure.interpolate(0, name="p_kp1")
    
    phikm1 = space_pressure.interpolate(0, name="phi_km1")
    phik = space_pressure.interpolate(0, name="phi_k")
    phikp1 = space_pressure.interpolate(0, name="phi_kp1")
    
    # u asterix
    ua = 2 * uk - ukm1
    # p hashtag
    ph = pk + 4/3 * phik - 1/3 * phikm1
    
    velocity_form = (
        inner(rho / (2 * dt) * (3 * u - 4 * uk + ukm1), v) * dx
        + inner(rho * grad(u) * ua + rho/2 * div(ua) * u, v) * dx
        + inner(mu * (grad(u) + nabla_grad(u)), grad(v)) * dx
        + inner(grad(ph), v) * dx
        == inner(f(t + dt), v) * dx
    )
    projection_form = (
        - inner(grad(phi), grad(q)) * dx
        == inner(3 * rho / (2 * dt) * div(ukp1), q) * dx
    )
    pressure_correction_form = (
        inner(p, q) * dx == inner(pk + phikp1, q) * dx
    )

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
        "newton.linear.tolerance": 1e-7,
        #"newton.linear.tolerance.strategy": "eisenstatwalker",
        #"newton.linear.errormeasure": "residualreduction",
        #"newton.linear.preconditioning.method": "ssor",
        "newton.verbose": True,
        "newton.linear.verbose": True
    }

    velocity_problem = galerkin(
        [velocity_form, *u_bc],
        solver="gmres",
        parameters={
            # slower than without
            #"newton.linear.preconditioning.method": "jacobi",
            #"newton.linear.preconditioning.method": "amg-ilu",
            **shared_solver_parameters
        }
    )
    projection_problem = galerkin(
        [projection_form, *p_bc],
        solver="cg",
        parameters={
            # slower than without
            #"newton.linear.preconditioning.method": "jacobi",
            **shared_solver_parameters
        }
    )
    pressure_problem = galerkin(
        [pressure_correction_form, *p_bc],
        solver="gmres",
        parameters={
            # slower than without
            #"newton.linear.preconditioning.method": "sor",
            **shared_solver_parameters
        }
    ) 
    
    def step(info_collection):
        info_collection.solve_event("velocity")(
            velocity_problem.solve(target=ukp1)
        )
        info_collection.solve_event("correction")(
            projection_problem.solve(target=phikp1)
        )
        info_collection.solve_event("pressure")(
            pressure_problem.solve(target=pkp1)
        )
        
        ukm1.assign(uk)
        uk.assign(ukp1)
        phikm1.assign(phik)
        phik.assign(phikp1)
        pk.assign(pkp1)
        t.value += dt.value

    def adapt():
        fem.adapt([ukm1, uk, pk, phikm1, phik])
        fem.loadBalance([ukm1, uk, pk, phikm1, phik])
        
    yield (ukm1, uk, pk, phik, step, adapt)

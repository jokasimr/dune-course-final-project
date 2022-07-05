import argparse
from pprint import pprint

parser = argparse.ArgumentParser(description='Run Karman Vortex experiment')
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--adaptive",  action='store_true')
parser.add_argument("--adaptivitytol", type=int, nargs=2, default=(50, 20))
parser.add_argument("--tentative", nargs="+", type=str, default=())
parser.add_argument("--pressure", nargs="+", type=str, default=())
parser.add_argument("--update", nargs="+", type=str, default=())
parser.add_argument("--mu", type=float, default=1e-3)
parser.add_argument("--dt", type=float, default=5e-5)
args = parser.parse_args()

def parse(v):
    if all(s in '0123456789.' for s in v):
        return float(v) if '.' in v else int(v)
    return v

def pairs(a):
    ps = []
    for i, x in enumerate(a):
        if i % 2 == 0:
            ps.append(x)
        else:
            ps.append(parse(x))
            yield ps
            ps = []

solver_arguments = {
    step: dict(list(pairs(getattr(args, step))))
    for step in ["tentative", "pressure", "update"]
}

import json
import numpy as np
import time

from dune.grid import cartesianDomain
from dune.fem.function import integrate, uflFunction, gridFunction
from dune.ufl import Constant
from dune.common import comm
from ufl import as_vector, sqrt, dot, curl, conditional, div
import dune.fem as fem

from info import RunInformationCollector
from chorins_method import setup

# Define domain
L = 2.2; H = 0.41
c = 0.2; r = 0.05

try:
    import pygmsh
    found_pygmsh = True

except Exception:
    found_pygmsh = False
    print("Could not import pygmsh")

if found_pygmsh and comm.rank == 0:
    # Grid size function
    def size(dim, tag, x, y, z, lc):
        d = ((x - c)**2 + (y - c)**2)**0.5
        if d > 3 * r:
            if x < 0.7:
                if 0.4 > x > 0.2:
                    return 0.025
                return 0.04
            return 0.06
        return abs(d - r)/2 + 1.5e-3
    
    with pygmsh.occ.Geometry() as geom:
        rectangle = geom.add_rectangle([0, 0, 0], L, H)
        obstacle = geom.add_ball([c, c, 0], r)
        geom.boolean_difference(rectangle, obstacle)
        geom.set_mesh_size_callback(size)
        mesh = geom.generate_mesh()
    
        points, cells = mesh.points, mesh.cells_dict
        domain = {"vertices": points[:,:2].astype(float),
                  "simplices": cells["triangle"].astype(int)}

        with open("domain.json", "w") as f:
            json_domain = {k: list(list(map(float, e)) for e in v) for k, v in domain.items()}
            json.dump(json_domain, f)

elif comm.rank == 0:
    with open("domain.json") as f:
        domain = json.load(f)

    domain = {k: np.array(v) for k, v in domain.items()}
    domain["simplices"] = domain["simplices"].astype(int)

else:
    domain = {"vertices": [], "simplices": []}

    
problem = setup(domain)
view, x = next(problem)
view.hierarchicalGrid.globalRefine(args.level)

# Define parameters
μ = Constant(args.mu, name="mu")
ρ = Constant(1, name="rho")
f = Constant([0, 0], name="source")

reynolds_number = 6 * 0.2 * (H - 0.2) / H**2 * 2*r / μ.value

T = 5
dt = Constant(args.dt, name="dt")
t = Constant(0, name="time")

# Boundary conditions
left = x[0] <= 0
right = x[0] >= L
top = x[1] >= H
bottom = x[1] <= 0
ball = dot((x - as_vector([c, c])), (x - as_vector([c, c]))) <= r**2 

bc_velocity = {top: [0, 0],
               bottom: [0, 0],
               ball: [0, 0],
               left: [6 * x[1] * (H - x[1]) / H**2, 0]}
bc_pressure = {right: 0}

u, p, *funs, step = problem.send((bc_velocity, bc_pressure, μ, ρ, t, dt, f, solver_arguments))

# Initial condition
u.interpolate([0, 0])
p.interpolate(0)

# Adaptivity
ignore_factor = Constant(1.2, name="ignore_factor")
indicator = conditional(
    dot((x - as_vector([c, c])), (x - as_vector([c, c]))) <= ignore_factor**2 * r**2,
    1,
    sqrt(curl(u)**2)
)
def adapt():
    fem.markNeighbors(
        indicator,
        refineTolerance=args.adaptivitytol[0],
        coarsenTolerance=args.adaptivitytol[1],
        minLevel=0,
        maxLevel=args.level)
    fem.adapt([u, p])
    fem.loadBalance([u, p, *funs])
    
indicator_function = uflFunction(view, ufl=indicator, name="refinement_indicator", order=2)
divergence_function = uflFunction(view, ufl=sqrt(div(u)**2), name="divergence", order=2)


@gridFunction(view, name="rank", order=0)
def rank_indicator(elem, x):
    return comm.rank

# Run
savetime = 0.01
write_solution = view.sequencedVTK(
    "solution",
    celldata=[indicator_function, divergence_function, rank_indicator],
    pointdata=[u, p],
    subsampling=2
)
use_adaptivity = args.adaptive
info = RunInformationCollector(
        vectors=dict(p=p, u=u),
        reynolds_number=reynolds_number,
        solver_arguments=solver_arguments,
        **vars(args)
)
if comm.rank == 0:
    pprint(info.export())
info.step_event(t.value)

while t.value < T:
    if t.value // savetime > (t.value - dt.value) // savetime:
        info.save()
        write_solution()
    step(info)
    print("Time: ", t.value)
    info.step_event(t.value)
    if use_adaptivity:
        adapt()
        info.adaptivity_event()

info.save()
write_solution()

import json
import numpy as np
import time

from dune.grid import cartesianDomain
from dune.fem.function import integrate, uflFunction
from dune.ufl import Constant
from dune.common import comm
from ufl import as_vector, sqrt, dot, curl, conditional, div
import dune.fem as fem

from info import RunInformationCollector
from pressure_correction import setup


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
                if 0.1 > x > 0.2:
                    return 0.02
                return 0.03
            return 0.05
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
#view.hierarchicalGrid.globalRefine(1)

# Define parameters
μ = Constant(1e-3, name="mu")
ρ = Constant(1, name="rho")
f = Constant([0, 0], name="source")

print("Reynolds number:", 6 * 0.2 * (H - 0.2) / H**2 * 2*r / μ.value)

T = 5
dt = Constant(5e-5, name="dt")
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

u0, u, p, phi, step, adapt_vectors = problem.send((bc_velocity, bc_pressure, μ, ρ, t, dt, f))

# Initial condition
u0.interpolate([0, 0])
u.interpolate([0, 0])
p.interpolate(0)
phi.interpolate(0)

# Adaptivity
ignore_factor = Constant(1.2, name="ignore_factor")
indicator = conditional(
    dot((x - as_vector([c, c])), (x - as_vector([c, c]))) <= ignore_factor**2 * r**2,
    1,
    sqrt(curl(u)**2)
)
def adapt():
    fem.markNeighbors(indicator, refineTolerance=50, coarsenTolerance=20, minLevel=0, maxLevel=3)
    adapt_vectors()
    
indicator_function = uflFunction(view, ufl=indicator, name="refinement_indicator", order=2)
divergence_function = uflFunction(view, ufl=sqrt(div(u)**2), name="divergence", order=2)

# Run
savetime = 0.01
write_solution = view.sequencedVTK(
    "solution",
    celldata=[indicator_function, divergence_function],
    pointdata=[u, p],
    subsampling=2
)
use_adaptivity = False
info = RunInformationCollector("vortex_street_pressure_correction", dict(p=p, u=u))
info.step_event(t.value)

while t.value < T:
    if t.value // savetime > (t.value - dt.value) // savetime:
        write_solution()
        info.save()
        if comm.rank == 0:
            print("JSON ", json.dumps(info.events[-1]))
        
    step(info)
    info.step_event(t.value)
    if use_adaptivity:
        adapt()
        info.adaptivity_event()

write_solution()
info.save()

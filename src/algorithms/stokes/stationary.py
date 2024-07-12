from firedrake import *
from typing import TypeAlias, Callable, Optional

from src.discretisation.space import SpaceDiscretisation

### abstract stationary Stokes algorithms
StationaryStokesAlgorithm: TypeAlias = Callable[[SpaceDiscretisation,Optional[Function]],tuple[Function,Function]]

### implementation
def mixedFEM(space_disc: SpaceDiscretisation,
             forcing: Function | None = None
             ) -> tuple[Function, Function]:
    """Solve stationary Stokes system with mixed finite elements. 
    
    Return velocity, pressure. """

    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    if not forcing:
        forcing, _ = Function(space_disc.mixed_space).subfunctions

    a = ( inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) )*dx
    L = ( inner(forcing,v) )*dx

    up = Function(space_disc.mixed_space)
    velocity, pressure = up.subfunctions

    solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
    
    #Mean correction
    mean_pressure = Constant(assemble( inner(pressure,1)*dx ))
    pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_pressure).dat.data
    
    return velocity, pressure


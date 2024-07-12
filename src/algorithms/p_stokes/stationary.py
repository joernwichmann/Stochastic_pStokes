from firedrake import *
import logging
from typing import TypeAlias, Callable, Optional

from src.discretisation.space import SpaceDiscretisation
from src.algorithms.nonlinearities import S_tensor
from src.algorithms.solver_configs import enable_monitoring

### abstract stationary Stokes algorithms
StationaryStokesAlgorithm: TypeAlias = Callable[[SpaceDiscretisation,Optional[float],Optional[float],Optional[Function]],tuple[Function,Function]]

### implementation
def mixedFEM(space_disc: SpaceDiscretisation,
             p_value: float = 2.0,
             kappa_value: float = 2.0,
             forcing: Function | None = None,
    ) -> tuple[Function,Function]:
    """Solve stationary p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return velocity, pressure."""
    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solve
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    v, q = TestFunctions(space_disc.mixed_space)

    if not forcing:
        forcing, _ = Function(space_disc.mixed_space).subfunctions

    VariationalForm = ( inner(S_tensor(grad(u),p_value,kappa_value), grad(v)) - inner(p, div(v)) + inner(div(u), q) - inner(forcing,v) )*dx

    #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
    try:
        solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
    except ConvergenceError as e:
        logging.exception(e)
        solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

    #correct pressure mean
    mean_p = Constant(assemble( inner(p,1)*dx ))

    pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data
    
    return velocity, pressure
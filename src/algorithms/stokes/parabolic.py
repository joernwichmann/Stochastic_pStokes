from firedrake import *
from copy import deepcopy
from tqdm import tqdm
from typing import TypeAlias, Callable, Optional

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation
from src.algorithms.solver_configs import enable_light_monitoring, direct_solve

### abstract structure of a Stokes algorithm
StokesAlgorithm: TypeAlias = Callable[
    [SpaceDiscretisation, list[float], list[float], Function, Function, Optional[dict[float,Function]], Optional[float]],
    tuple[dict[float,Function],dict[float,Function]]
]

### converter that maps a string representation of the algorithm to its implementation
def get_algorithm_by_name(algorithm_name: str) -> StokesAlgorithm:
    match algorithm_name:
        case "Chorin splitting":
            return Chorin_splitting
        case "Implicit Euler mixed FEM":
            return implicitEuler_mixedFEM
        case "Crank Nicolson mixed FEM Stratonovich Transport Noise":
            return CrankNicolson_mixedFEM_strato_transportNoise
        case "Crank Nicolson mixed FEM Stratonovich Transport Noise asymmetric":
            return CrankNicolson_mixedFEM_strato_transportNoise_asym
        case "Implicit Euler mixed FEM Ito Transport Noise":
            return impliciteEuler_mixedFEM_ito_transportNoise
        case "Implicit Euler mixed FEM Stratonovich Transport Noise asymmetric":
            return impliciteEuler_mixedFEM_strato_transportNoise_asym
        case "Theta Scheme mixed FEM Stratonovich Transport Noise asymmetric":
            return ThetaScheme_mixedFEM_strato_transportNoise_asym
        case other:
            print(f"The algorithm '{algorithm_name}' is not avaiable.")
            raise NotImplementedError

### implementations of abstract structure
def implicitEuler_mixedFEM(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with mixed finite elements. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    uold.assign(initial_condition)

    a = ( inner(u,v) + tau*( 1.0/Re*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) ) )*dx
    L = ( inner(uold,v) + tau*inner(det_forcing,v) + dW*inner(noise_coefficient, v) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

 
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
            
        solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)

        #Mean correction
        mean_p = Constant(assemble( inner(p,1)*dx ))
        p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(u)
        time_to_pressure[time] = deepcopy(p)

        upold.assign(up)

    return time_to_velocity, time_to_pressure

def Chorin_splitting(space_disc: SpaceDiscretisation,
                     time_grid: list[float],
                     noise_steps: list[float],
                     noise_coefficient: Function,
                     initial_condition: Function,
                     time_to_det_forcing: dict[float,Function] | None = None,
                     Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with Chorin splitting. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)
    det_forcing = Function(space_disc.velocity_space)

    u = TrialFunction(space_disc.velocity_space)
    v = TestFunction(space_disc.velocity_space)
    uold = Function(space_disc.velocity_space)
    unew = Function(space_disc.velocity_space)
    utilde = Function(space_disc.velocity_space)

    p = TrialFunction(space_disc.pressure_space)
    q = TestFunction(space_disc.pressure_space)
    pnew = Function(space_disc.pressure_space)

    a1 = ( inner(u,v) + tau*inner(grad(u), grad(v)) )*dx
    L1 = ( inner(uold,v) + tau*inner(det_forcing,v) + dW*inner(noise_coefficient, v) )*dx

    a2 = inner(grad(p),grad(q))*dx
    L2 = 1/tau*inner(utilde,grad(q))*dx

    a3 = inner(u,v)*dx
    L3 = ( inner(utilde,v) - tau*inner(grad(pnew),v) )*dx

    V_basis = VectorSpaceBasis(constant=True)

    uold.assign(initial_condition)

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

 
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pnew)

    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        solve(a1 == L1, utilde, bcs=space_disc.bcs_vel)
        solve(a2 == L2, pnew, nullspace = V_basis)
        solve(a3 == L3, unew)

        #Mean correction
        mean_p = Constant(assemble( inner(pnew,1)*dx ))
        pnew.dat.data[:] = pnew.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(unew)
        time_to_pressure[time] = deepcopy(pnew)

        uold.assign(unew)

    return time_to_velocity, time_to_pressure


def impliciteEuler_mixedFEM_ito_transportNoise(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with mixed finite elements. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    uold.assign(initial_condition)

    a = ( inner(u,v) + tau*( 1.0/Re*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) ) )*dx
    L = ( inner(uold,v) + tau*inner(det_forcing,v) + dW*inner(dot(grad(uold), noise_coefficient), v) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

 
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
            
        solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)

        #Mean correction
        mean_p = Constant(assemble( inner(p,1)*dx ))
        p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(u)
        time_to_pressure[time] = deepcopy(p)

        upold.assign(up)

    return time_to_velocity, time_to_pressure

def impliciteEuler_mixedFEM_strato_transportNoise_asym(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with mixed finite elements. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    uold.assign(initial_condition)

    a = ( inner(u,v) + tau*( 1.0/Re*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) ) - dW/4.0*(inner(dot(grad(u),noise_coefficient), v) - inner(dot(grad(v),noise_coefficient), u)) )*dx
    L = ( inner(uold,v) + tau*inner(det_forcing,v) + dW/4.0*(inner(dot(grad(uold),noise_coefficient), v) - inner(dot(grad(v),noise_coefficient), uold)) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

 
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):

        dW.assign(noise_steps[index])
        #dW.assign(np.sqrt(time_increments[index]))
        tau.assign(time_increments[index])
        time += time_increments[index]
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
            
        solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null) 

        ###check various terms
        check_cancel = assemble( dW/4.0*( inner(dot(grad(u),noise_coefficient), u) - inner(dot(grad(u),noise_coefficient), u) )*dx )
        check_div = assemble( inner(p, div(u))*dx )
        print(f"time = {time}\ncheck_cancel = {check_cancel:.2E}\ncheck_div = {check_div:.2E}")

        #Mean correction
        mean_p = Constant(assemble( inner(p,1)*dx ))
        p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(u)
        time_to_pressure[time] = deepcopy(p)

        upold.assign(up)

    return time_to_velocity, time_to_pressure

def CrankNicolson_mixedFEM_strato_transportNoise(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with mixed finite elements. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    uold.assign(initial_condition)

    
    a = ( inner(u,v) + tau*( 1.0/(2.0*Re)*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) ) - dW/2.0*inner(dot(grad(u),noise_coefficient), v) )*dx
    L = ( inner(uold,v) + tau*( -1.0/(2.0*Re)*inner(grad(uold), grad(v)) + inner(det_forcing,v) ) + dW/2.0*inner(dot(grad(uold),noise_coefficient), v) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

 
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
            
        #solve for u(n+1)
        solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)

        #Mean correction
        mean_p = Constant(assemble( inner(p,1)*dx ))
        p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        time_to_velocity[time] = deepcopy(u)
        time_to_pressure[time] = deepcopy(p)

        upold.assign(up)

    return time_to_velocity, time_to_pressure

def CrankNicolson_mixedFEM_strato_transportNoise_asym(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with mixed finite elements. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    uold.assign(initial_condition)

    a = ( inner(u,v) + tau/2.0*( 1.0/Re*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) ) - dW/4.0*( inner(dot(grad(u),noise_coefficient), v) - inner(dot(grad(v),noise_coefficient), u) ) )*dx
    L = ( inner(uold,v) + tau/2.0*( inner(det_forcing,v) ) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

 
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        #set time step parameters
        dW.assign(noise_steps[index])
        #dW.assign(np.sqrt(time_increments[index]))
        tau.assign(time_increments[index])
        time += time_increments[index]

        #load det forcing if needed
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
            
        #solve for u(n+1/2)    
        solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null,solver_parameters=enable_light_monitoring)

        #extrapolation to obtain u(n+1)
        check_div_half = assemble( inner(p, div(u))*dx )
        u.dat.data[:] = 2*u.dat.data - uold.dat.data
        #p.dat.data[:] = 2*p.dat.data - pold.dat.data

        ###check various terms
        check_cancel = assemble( dW/4.0*( inner(dot(grad(u),noise_coefficient), u) - inner(dot(grad(u),noise_coefficient), u) )*dx )
        check_div = assemble( inner(p, div(u))*dx )
        print(f"time = {time}\ncheck_cancel = {check_cancel:.2E}\ncheck_div = {check_div:.2E}\ncheck_div_halftime = {check_div_half:.2E}")


        #Mean correction
        mean_p = Constant(assemble( inner(p,1)*dx ))
        p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data


        time_to_velocity[time] = deepcopy(u)
        time_to_pressure[time] = deepcopy(p)

        upold.assign(up)

    return time_to_velocity, time_to_pressure


def ThetaScheme_mixedFEM_strato_transportNoise_asym(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1,
                           theta: float = 4/8.0) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve Stokes system with mixed finite elements. 
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """

    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)
    theta_const = Constant(theta)

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    uold.assign(initial_condition)

    a = ( inner(u,v) + theta_const*tau*( 1.0/Re*inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) ) - theta_const*dW/2.0*( inner(dot(grad(u),noise_coefficient), v) - inner(dot(grad(v),noise_coefficient), u) ) )*dx
    L = ( inner(uold,v) + theta_const*tau*( inner(det_forcing,v) ) +(theta_const - 1/2.0)*dW/2.0*( inner(dot(grad(uold),noise_coefficient), v) - inner(dot(grad(v),noise_coefficient), uold) ) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

 
    time_to_velocity = dict()
    time_to_pressure = dict()

    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        #set time step parameters
        dW.assign(noise_steps[index])
        #dW.assign(np.sqrt(time_increments[index]))
        tau.assign(time_increments[index])
        time += time_increments[index]

        #load det forcing if needed
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
            
        #solve for u(n+theta)    
        solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null,solver_parameters=enable_light_monitoring)

        #extrapolation to obtain u(n+1)
        print(f"theta = {theta}")
        check_div_half = assemble( inner(p, div(u))*dx )
        u.dat.data[:] = 1/theta*u.dat.data + (1 - 1/theta)*uold.dat.data

        ###check various terms
        check_cancel = assemble( dW/4.0*( inner(dot(grad(u),noise_coefficient), u) - inner(dot(grad(u),noise_coefficient), u) )*dx )
        check_div = assemble( inner(p, div(u))*dx )
        print(f"time = {time}\ncheck_cancel = {check_cancel:.2E}\ncheck_div = {check_div:.2E}\ncheck_div_{theta}-time = {check_div_half:.2E}")


        #Mean correction
        mean_p = Constant(assemble( inner(p,1)*dx ))
        p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data


        time_to_velocity[time] = deepcopy(u)
        time_to_pressure[time] = deepcopy(p)

        upold.assign(up)

    return time_to_velocity, time_to_pressure




from firedrake import *
from copy import deepcopy
from tqdm import tqdm
from typing import TypeAlias, Callable, Optional
import logging

from src.discretisation.time import trajectory_to_incremets
from src.discretisation.space import SpaceDiscretisation
from src.algorithms.nonlinearities import S_tensor, S_tensor_sym, epsilon
from src.algorithms.solver_configs import enable_monitoring, direct_solve_details

### abstract structure of a p-Stokes algorithm
pStokesAlgorithm: TypeAlias = Callable[
    [SpaceDiscretisation, list[float], list[float], Function, Function, Optional[float], Optional[float], Optional[dict[float,Function]], Optional[float]],
    tuple[dict[float,Function],dict[float,Function]]
]

### converter that maps a string representation of the algorithm to its implementation   
def get_algorithm_by_name(algorithm_name: str) -> pStokesAlgorithm:
    match algorithm_name:
        case "Implicit Euler mixed FEM":
            return implicitEuler_mixedFEM
        case "Implicit Euler mixed FEM linear multi Noise":
            return implicitEuler_mixedFEM_linearMulti
        case "Implicit Euler mixed FEM linear multi Noise with sym Grad":
            return implicitEuler_mixedFEM_linearMulti_withSymGrad
        case "Implicit Euler mixed FEM linear multi Noise with sym Grad and approx of Averages":
            return implicitEuler_mixedFEM_linearMulti_withSymGrad_approxOfAverages
        case "Crank Nicolson mixed FEM Stratonovich Transport Noise":
            return CrankNicolson_mixedFEM_strato_transportNoise
        case "Crank Nicolson mixed FEM Stratonovich Transport Noise with Temam Symmetrisation":
            return CrankNicolson_mixedFEM_strato_transportNoise_withTemamsym
        case "Implicit Euler mixed FEM Ito Transport Noise":
            return impliciteEuler_mixedFEM_ito_transportNoise
        case "Implicit Euler mixed FEM Stratonovich Transport Noise":
            return impliciteEuler_mixedFEM_strato_transportNoise
        case "Implicit Euler mixed FEM Stratonovich Transport Noise with Temam Symmetrisation":
            return impliciteEuler_mixedFEM_strato_transportNoise_withTemamsym
        case "Crank Nicolson mixed FEM Stratonovich Transport Noise with anti-symmetrisation":
            return CrankNicolson_mixedFEM_strato_transportNoise_withAntisym
        case other:
            print(f"The algorithm '{algorithm_name}' is not avaiable.")
            raise NotImplementedError

### implementations of abstract structure
def implicitEuler_mixedFEM(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor(grad(u),p_value,kappa_value), grad(v)) - inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - dW*inner(noise_coefficient, v)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        time_to_pressure[time] = deepcopy(pressure)

        #update uold to proceed time-steppping
        uold.assign(velocity)

    return time_to_velocity, time_to_pressure

def implicitEuler_mixedFEM_linearMulti(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1,
                           noise_intensity: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor(grad(u),p_value,kappa_value), grad(v)) - inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - noise_intensity*dW*inner(uold, v)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        time_to_pressure[time] = deepcopy(pressure)

        #update uold to proceed time-steppping
        uold.assign(velocity)

    return time_to_velocity, time_to_pressure

def implicitEuler_mixedFEM_linearMulti_withSymGrad(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1,
                           noise_intensity: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor_sym(grad(u),p_value,kappa_value), epsilon(grad(v))) - inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - noise_intensity*dW*inner(uold, v)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        time_to_pressure[time] = deepcopy(pressure)

        #update uold to proceed time-steppping
        uold.assign(velocity)

    return time_to_velocity, time_to_pressure

def implicitEuler_mixedFEM_linearMulti_withSymGrad_approxOfAverages(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1,
                           noise_intensity: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    ###intermediate solution
    upint = Function(space_disc.mixed_space)
    uint, pint = upint.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition
    uold.assign(initial_condition)
    uint.assign(initial_condition)

    #meshpara
    xcoord, ycoord = SpatialCoordinate(space_disc.mesh)

    # build variational form
    VariationalForm = ( 
        inner(u - uint,v) 
        + tau*1.0/Re*inner( S_tensor_sym(grad(u),p_value,kappa_value), epsilon(grad(v)))
        + tau*(- inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - noise_intensity*dW*inner(uold, v)
        - dW*inner(noise_coefficient, v)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            #logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=direct_solve_details)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        time_to_pressure[time] = deepcopy(pressure)

        #update old and intermediate solution to proceed time-steppping
        if index >= 1:
            uold.assign(uint)
        uint.assign(velocity)

    return time_to_velocity, time_to_pressure

def CrankNicolson_mixedFEM_strato_transportNoise(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor((grad(u) + grad(uold))/2.0,p_value,kappa_value), grad(v)) - inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - dW/2.0*inner(dot(grad(u) + grad(uold), noise_coefficient), v)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        time_to_pressure[time] = deepcopy(pressure)

        #update uold to proceed time-steppping
        uold.assign(velocity)

    return time_to_velocity, time_to_pressure

def CrankNicolson_mixedFEM_strato_transportNoise_withTemamsym(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor((grad(u) + grad(uold))/2.0,p_value,kappa_value), grad(v)) - inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - dW/2.0*inner(dot(grad(u) + grad(uold), noise_coefficient), v)
        - dW/4.0*inner(div(noise_coefficient)* (u + uold), v)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        time_to_pressure[time] = deepcopy(pressure)

        #update uold to proceed time-steppping
        uold.assign(velocity)

    return time_to_velocity, time_to_pressure


def CrankNicolson_mixedFEM_strato_transportNoise_withAntisym(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor((grad(u) + grad(uold))/2.0,p_value,kappa_value), grad(v)) - inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - dW/4.0*inner(dot(grad(u) + grad(uold), noise_coefficient), v)
        + dW/4.0*inner(dot(grad(v), noise_coefficient), u + uold)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()
    time_to_velocity_midpoints = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_velocity_midpoints[time] = deepcopy(det_forcing)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        velocity_mid = Function(space_disc.velocity_space)
        velocity_mid.dat.data[:] = (velocity.dat.data + uold.dat.data)/2.0
        time_to_velocity_midpoints[time] = deepcopy(velocity_mid)
        time_to_pressure[time] = deepcopy(pressure)

        #update uold to proceed time-steppping
        uold.assign(velocity)

    return time_to_velocity, time_to_pressure, time_to_velocity_midpoints


def impliciteEuler_mixedFEM_ito_transportNoise(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor(grad(u),p_value,kappa_value), grad(v)) - inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - dW*inner(dot(grad(uold), noise_coefficient), v)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        time_to_pressure[time] = deepcopy(pressure)

        #update uold to proceed time-steppping
        uold.assign(velocity)

    return time_to_velocity, time_to_pressure


def impliciteEuler_mixedFEM_strato_transportNoise(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor(grad(u),p_value,kappa_value), grad(v)) - inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - dW/2.0*inner(dot(grad(u) + grad(uold), noise_coefficient), v)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        time_to_pressure[time] = deepcopy(pressure)

        #update uold to proceed time-steppping
        uold.assign(velocity)

    return time_to_velocity, time_to_pressure

def impliciteEuler_mixedFEM_strato_transportNoise_withTemamsym(space_disc: SpaceDiscretisation,
                           time_grid: list[float],
                           noise_steps: list[float], 
                           noise_coefficient: Function,
                           initial_condition: Function,
                           p_value: float = 2.0,
                           kappa_value: float = 0.1,
                           time_to_det_forcing: dict[float,Function] | None = None, 
                           Reynolds_number: float = 1) -> tuple[dict[float,Function], dict[float,Function]]:
    """Solve p-Stokes system with kappa regularisation and mixed finite elements. The viscous stress is given by S(A) = (kappa + |A|^2)^((p-2)/2)A.
    
    Return 'time -> velocity' and 'time -> pressure' dictionaries. """
    # initialise constants in variational form
    Re = Constant(Reynolds_number)
    tau = Constant(1.0)
    dW = Constant(1.0)

    # initialise function objects
    v, q = TestFunctions(space_disc.mixed_space)

    up = Function(space_disc.mixed_space)
    u, p = split(up)    # split types: <class 'ufl.tensors.ListTensor'> and <class 'ufl.indexed.Indexed'> needed for nonlinear solver
    velocity, pressure = up.subfunctions    #subfunction types: <class 'firedrake.function.Function'> and <class 'firedrake.function.Function'>

    upold = Function(space_disc.mixed_space)
    uold, pold = upold.subfunctions

    # initialise deterministic forcing by zero as default 
    det_forcing, _ = Function(space_disc.mixed_space).subfunctions

    # set initial condition to uold
    uold.assign(initial_condition)

    # build variational form
    VariationalForm = ( 
        inner(u - uold,v) 
        + tau*( 1.0/Re*inner( S_tensor(grad(u),p_value,kappa_value), grad(v)) - inner(p, div(v)) + inner(div(u), q) )
        - tau*inner(det_forcing,v)
        - dW/2.0*inner(dot(grad(u) + grad(uold), noise_coefficient), v)
        - dW/4.0*inner(div(noise_coefficient)* (u + uold), v)
        )*dx

    # setup initial time and time increments
    initial_time, time_increments = trajectory_to_incremets(time_grid)
    time = initial_time

    # initialise storage for solution output
    time_to_velocity = dict()
    time_to_pressure = dict()

    # store initialisation of time-stepping
    time_to_velocity[time] = deepcopy(uold)
    time_to_pressure[time] = deepcopy(pold)

    #check if deterministic and random increments are iterables of the same length
    if not len(time_increments) == len(noise_steps):
        msg_error = "Time grid and noise grid are not of the same length.\n"
        msg_error += f"Time grid length: \t {len(time_increments)}\n"
        msg_error += f"Noise grid length: \t {len(noise_steps)}"
        raise ValueError(msg_error)

    for index in tqdm(range(len(time_increments))):
        # update random and deterministc time step, and nodal time
        dW.assign(noise_steps[index])
        tau.assign(time_increments[index])
        time += time_increments[index]
        
        # if provided change deterministic forcing to provided one
        if time_to_det_forcing:
            try:
                det_forcing.assign(time_to_det_forcing[time])
            except KeyError as k:
                print(f"Deterministic forcing couldn't be set.\nRequested time:\t {time}\nAvailable times:\t {list(time_to_det_forcing.keys())}")
                raise k
        
        #try solve nonlinear problem by using firedrake blackbox. If default solve doesn't converge, restart solve with enbabled montoring to see why it fails. 
        try:   
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)
        except ConvergenceError as e:
            logging.exception(e)
            solve(VariationalForm == 0, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null, solver_parameters=enable_monitoring)

        #correct mean-value of pressure
        mean_p = Constant(assemble( inner(p,1)*dx ))
        pressure.dat.data[:] = pressure.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

        #store solution
        time_to_velocity[time] = deepcopy(velocity)
        time_to_pressure[time] = deepcopy(pressure)

        #update uold to proceed time-steppping
        uold.assign(velocity)

    return time_to_velocity, time_to_pressure
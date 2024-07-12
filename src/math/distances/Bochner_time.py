"""Defines Bochner time distances."""
from numpy import sqrt
from typing import Callable, TypeAlias
from firedrake import Function

from src.math.distances.space import SpaceDistance
from src.math.norms.Bochner_time import nikolskii_half_X_norm
from src.math.norms.space import SpaceNorm

#############################           BOCHNER TIME DISTANCE
#abstract concept
BochnerTimeDistance: TypeAlias = Callable[[dict[float,Function],dict[float,Function],SpaceDistance],float]

#utility
def project_left(time: float, time_grid: list[float]) -> float:
    """Return the biggest nodal time in the time grid that is not bigger than the requested time."""
    sorted_time = sorted(time_grid)
    if time < sorted_time[0]:
        raise ValueError("Requested time is not in the time_grid.")
    for node in sorted_time:
        if node > time:
            return projected_time
        projected_time = node
    return projected_time

def integrate_in_time(time_to_function: dict[float, Function]) -> dict[float, Function]:
    sorted_time = sorted(list(time_to_function.keys()))
    time_to_int_function = dict()
    time_to_int_function[sorted_time[0]] = time_to_function[sorted_time[0]]
    for k in range(1,len(sorted_time)):
        time_to_int_function[sorted_time[k]] = time_to_function[sorted_time[k]]*(sorted_time[k] - sorted_time[k-1])
    return time_to_int_function


#implementation
def linf_X_distance(time_to_function1: dict[float, Function], time_to_function2: dict[float,Function], distance_X: SpaceDistance) -> float:
    """Computes the Linf in time and 'X' in space distance of 'time -> function1' and 'time -> function2' dictionaries. 
    
    In more detail: The time grids of function 1 and function 2 are used to generate a unified time grid.
    Local errors on the unified time grid are computed by comparing the distance of function 1 
    and function 2 measured in 'X' at time points that are the biggest nodal times below the unified nodal time
    in time grid 1 and time grid 2, respectively. The biggest local error is returned."""
    time_grid1 = set(time_to_function1.keys())
    time_grid2 = set(time_to_function2.keys())
    union_time = time_grid1.union(time_grid2)

    #compute local errors and store them in a dictionary indexed by the time
    error = []
    for time in union_time:
        time_projected_grid1 = project_left(time,time_grid1)
        time_projected_grid2 = project_left(time,time_grid2)
        error.append(distance_X(time_to_function1[time_projected_grid1], time_to_function2[time_projected_grid2]))
    return max(error)

def l2_X_distance(time_to_function1: dict[float, Function], time_to_function2: dict[float,Function], distance_X: SpaceDistance) -> float:
    """Computes the L2 in time and 'X' in space distance of 'time -> function1' and 'time -> function2' dictionaries. 
    
    In more detail: The time grids of function 1 and function 2 are used to generate a unified time grid.
    Local errors on the unified time grid are computed by comparing the distance of function 1 
    and function 2 measured in 'X' at time points that are the biggest nodal times below the unified nodal time
    in time grid 1 and time grid 2, respectively. 
    Afterwards the square of the local errors are weighted by increments of the unified time grid and summed up.
    Finally its square root is returned."""

    time_grid1 = set(time_to_function1.keys())
    time_grid2 = set(time_to_function2.keys())
    union_time = time_grid1.union(time_grid2)

    #compute local errors and store them in a dictionary indexed by the time
    error_grid = dict()
    for time in union_time:
        time_projected_grid1 = project_left(time,time_grid1)
        time_projected_grid2 = project_left(time,time_grid2)
        error_grid[time] = distance_X(time_to_function1[time_projected_grid1], time_to_function2[time_projected_grid2])

    #sum up the local error contributions weighted by the size of the local time steps
    sorted_union_time = sorted(list(union_time))
    error = 0
    for k, time in enumerate(sorted_union_time[1:]):
        error += error_grid[time]**2*(sorted_union_time[k+1] - sorted_union_time[k])
    return sqrt(error)

def end_time_X_distance(time_to_function1: dict[float, Function], time_to_function2: dict[float,Function], distance_X: SpaceDistance) -> float:
    """Computes the 'X' distance in space of 'time -> function1' and 'time -> function2' dictionaries at the endtime. """
    end_time1 = sorted(list(time_to_function1.keys()))[-1]
    end_time2 = sorted(list(time_to_function2.keys()))[-1]
    if not end_time1 == end_time2:
        msg_error = "End times do not match.\n"
        msg_error += f"First end time: \t {end_time1}\n"
        msg_error += f"Second end time: \t {end_time2}"
        raise ValueError(msg_error)
    return distance_X(time_to_function1[end_time1],time_to_function2[end_time2])

def h_minus1_X_distance(time_to_function1: dict[float, Function], time_to_function2: dict[float,Function], distance_X: SpaceDistance) -> float:
    """Computes the H-1 in time and 'X' in space distance of 'time -> function1' and 'time -> function2' dictionaries. 
    
    In more detail: The time grids of function 1 and function 2 are used to generate a unified time grid.
    Local errors on the unified time grid are computed by comparing the distance of function 1 
    and function 2 measured in 'X' at time points that are the biggest nodal times below the unified nodal time
    in time grid 1 and time grid 2, respectively. 
    Afterwards the square of the local errors are weighted by increments of the unified time grid and summed up.
    Finally its square root is returned."""

    #integrate input functions in time:
    time_to_int_function1 = integrate_in_time(time_to_function1)
    time_to_int_function2 = integrate_in_time(time_to_function2)

    return l2_X_distance(time_to_int_function1, time_to_int_function2, distance_X)

def w_minus1_inf_X_distance(time_to_function1: dict[float, Function], time_to_function2: dict[float,Function], distance_X: SpaceDistance) -> float:
    """Computes the W{-1,inf} in time and 'X' in space distance of 'time -> function1' and 'time -> function2' dictionaries. """
    
    #integrate input functions in time:
    time_to_int_function1 = integrate_in_time(time_to_function1)
    time_to_int_function2 = integrate_in_time(time_to_function2)

    return linf_X_distance(time_to_int_function1, time_to_int_function2, distance_X)

def nikolskii_minushalf_X_distance(time_to_function1: dict[float, Function], time_to_function2: dict[float,Function], norm_X: SpaceNorm) -> float:
    """Computes the N{-1/2,2} in time and 'X' in space distance of 'time -> function1' and 'time -> function2' dictionaries. 
    
    In more detail: The time grids of function 1 and function 2 are used to generate a unified time grid.
    Local errors on the unified time grid are computed by comparing the distance of function 1 
    and function 2 measured in 'X' at time points that are the biggest nodal times below the unified nodal time
    in time grid 1 and time grid 2, respectively. 
    Afterwards the square of the local errors are weighted by increments of the unified time grid and summed up.
    Finally its square root is returned."""

    #integrate input functions in time:
    time_to_int_function1 = integrate_in_time(time_to_function1)
    time_to_int_function2 = integrate_in_time(time_to_function2)

    #generate joint time grid
    time_grid1 = set(time_to_int_function1.keys())
    time_grid2 = set(time_to_int_function2.keys())
    union_time = time_grid1.union(time_grid2)

    #define difference function
    time_to_function_dif = dict()
    for time in union_time:
        time_to_function_dif[time] = time_to_int_function1[project_left(time,time_grid1)] - time_to_int_function2[project_left(time,time_grid2)]

    return nikolskii_half_X_norm(time_to_function_dif,norm_X)

    
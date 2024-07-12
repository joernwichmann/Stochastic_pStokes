"""Defines Bochner time norms."""
from numpy import sqrt
from firedrake import Function
from typing import Callable, TypeAlias

from src.math.norms.space import SpaceNorm

#############################           BOCHNER TIME NORMS
#abstract concept
BochnerTimeNorm: TypeAlias = Callable[[dict[float, Function],SpaceNorm],float]

#utility
def integrate_in_time(time_to_function: dict[float, Function]) -> dict[float, Function]:
    sorted_time = sorted(list(time_to_function.keys()))
    time_to_int_function = dict()
    time_to_int_function[sorted_time[0]] = time_to_function[sorted_time[0]]
    for k in range(1,len(sorted_time)):
        time_to_int_function[sorted_time[k]] = time_to_function[sorted_time[k]]*(sorted_time[k] - sorted_time[k-1])
    return time_to_int_function

#implementation
def linf_X_norm(time_to_function: dict[float, Function], norm_X: SpaceNorm) -> float:
    """Computes the Linf in time and 'X' in space norm of 'time -> function1' dictionary."""
    return max([norm_X(time_to_function[time]) for time in time_to_function.keys()])

def l2_X_norm(time_to_function: dict[float, Function], norm_X: SpaceNorm) -> float:
    """Computes the L2 in time and 'X' in space norm of 'time -> function1' dictionary."""
    norm_grid = {time: norm_X(time_to_function[time]) for time in time_to_function.keys()}
    #sum up the local error contributions weighted by the size of the local time steps
    sorted_time = sorted(list(time_to_function.keys()))
    norm = 0
    for k, time in enumerate(sorted_time[1:]):
        norm += norm_grid[time]**2*(sorted_time[k+1] - sorted_time[k])
    return sqrt(norm)

def end_time_X_norm(time_to_function: dict[float, Function], norm_X: SpaceNorm) -> float:
    """Computes the 'X' norm in space of 'time -> function1' dictionary at the endtime."""
    end_time = sorted(list(time_to_function.keys()))[-1]
    return norm_X(time_to_function[end_time])

def h_minus1_X_norm(time_to_function: dict[float, Function], norm_X: SpaceNorm) -> float:
    """Computes the H-1 in time and 'X' in space norm of 'time -> function1' dictionary."""
    time_to_int_function = integrate_in_time(time_to_function)
    return l2_X_norm(time_to_int_function,norm_X)

def nikolskii_half_X_norm(time_to_function: dict[float, Function], norm_X: SpaceNorm) -> float:
    """Computes the N{1/2,2} in time and 'X' in space norm of 'time -> function' dictionary."""
    error = 0
    sorted_time = sorted(list(time_to_function.keys()))
    for k in range(len(sorted_time)):
        local_error = 0
        for l in range(k,len(sorted_time)):
            local_error += norm_X(time_to_function[sorted_time[l]] -time_to_function[sorted_time[l-k]])**2
        if local_error > error:
            error = local_error
    return sqrt(error)
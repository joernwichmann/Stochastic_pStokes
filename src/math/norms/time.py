"""Defines time norms."""
from numpy import sqrt, absolute
from typing import Callable, TypeAlias

#############################           TIME NORMS
#abstract concept
TimeNorm: TypeAlias = Callable[[dict[float,float]],float]

#implementation
def linf_time(time_to_float: dict[float, float]) -> float:
    """Computes the Linf in time norm of 'time -> value' dictionary."""
    return max([absolute(time_to_float[time]) for time in time_to_float.keys()])

def l2_time(time_to_float: dict[float, float]) -> float:
    """Computes the L2 in time norm of 'time -> value' dictionary."""
    absolute_grid = {time: absolute(time_to_float[time]) for time in time_to_float.keys()}
    #sum up the local error contributions weighted by the size of the local time steps
    sorted_time = sorted(list(time_to_float.keys()))
    norm = 0
    for k, time in enumerate(sorted_time[1:]):
        norm += absolute_grid[time]**2*(sorted_time[k+1] - sorted_time[k])
    return sqrt(norm)

def l1_time(time_to_float: dict[float, float]) -> float:
    """Computes the L1 in time norm of 'time -> value' dictionary."""
    absolute_grid = {time: absolute(time_to_float[time]) for time in time_to_float.keys()}
    #sum up the local error contributions weighted by the size of the local time steps
    sorted_time = sorted(list(time_to_float.keys()))
    norm = 0
    for k, time in enumerate(sorted_time[1:]):
        norm += absolute_grid[time]*(sorted_time[k+1] - sorted_time[k])
    return norm

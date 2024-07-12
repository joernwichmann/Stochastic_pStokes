"""Define energies."""
from firedrake import Function
from typing import Callable, TypeAlias

from src.math.norms.space import l2_space, h1_space

#############################           ENERGIES 
#abstract concept
Energy_function: TypeAlias = Callable[[dict[float,Function]],dict[float,float]]

#implementation
def kinetic_energy(time_to_function: dict[float,Function]) -> dict[float,float]:
    """Compute the kinetic energy of a function."""
    return  {time: l2_space(time_to_function[time]**2/2.0) for time in time_to_function.keys()}

def potential_energy(time_to_function: dict[float,Function]) -> dict[float,float]:
    """Compute the potential energy of the function"""
    return {time: h1_space(time_to_function[time])**2 for time in time_to_function.keys()}

def accumulated_potential_energy(time_to_function: dict[float,Function]) -> dict[float,float]:
    """Compute the accumulated time-weighted potential energy of the function"""
    local_energy = potential_energy(time_to_function)
    #sum up the local error contributions weighted by the size of the local time steps
    sorted_time = sorted(list(time_to_function.keys()))
    accumulated_energy = {sorted_time[0]: 0}
    for k, time in enumerate(sorted_time[1:]):
        accumulated_energy[time] = accumulated_energy[sorted_time[k]] + local_energy[time]*(sorted_time[k+1] - sorted_time[k])
    return accumulated_energy
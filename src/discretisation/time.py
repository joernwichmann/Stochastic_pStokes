from dataclasses import dataclass
from functools import cached_property

from src.string_formatting import format_header

def _time_stepsize(initial_time: float, end_time: float, time_steps: int) -> float:
    """Return stepsize based on time parameter."""
    return (end_time - initial_time)/time_steps
    
def _time_steps(refinement_level: int) -> int:
    """Return number of steps based on time parameter."""
    return 2**refinement_level

def _time_grid(initial_time: float, end_time: float, time_steps: int) -> list[float]:
    """Return time grid based on time parameter."""
    return [initial_time + k*(end_time - initial_time)/time_steps for k in range(time_steps +1)]

@dataclass
class TimeDiscretisation:
    """Store time parameter."""
    initial_time: float
    end_time: float
    refinement_levels: list[int]

    @cached_property
    def ref_to_time_grid(self) -> dict[int,list[float]]:
         return {level: _time_grid(self.initial_time, self.end_time, _time_steps(level)) for level in self.refinement_levels }
    
    @cached_property
    def ref_to_time_steps(self) -> dict[int,int]:
         return {level: _time_steps(level) for level in self.refinement_levels}
    
    @cached_property
    def ref_to_time_stepsize(self) -> dict[int,float]:
         return {level: _time_stepsize(self.initial_time, self.end_time, _time_steps(level)) for level in self.refinement_levels }
    
    @cached_property
    def ref_to_time_to_id(self) -> dict[int,dict[float,int]]:
        return {level: {time: index for index, time in enumerate(self.ref_to_time_grid[level])} for level in self.refinement_levels}

    
    def __str__(self) -> str:
        out = format_header("TIME PARAMETER")
        out += f"\nInitial time: \t \t {self.initial_time}"
        out += f"\nEnd time: \t \t {self.end_time} \n"
        out += format_header("")
        out += "\n\t ref level \t | \t stepsize \t | \t no of steps \t"
        out += format_header("")
        for level in self.refinement_levels:
            out += f"\n\t {level} \t \t | \t {self.ref_to_time_stepsize[level]:.03f} \t \t | \t {_time_steps(level)} \t"
        return out
    

### utilities for time grids
def increments_to_trajectory(initial_condition: float, increments: list[float]) -> list[float]:
    """Return trajectory based on an initial condition and its increment updates."""
    trajectory = [initial_condition]
    for increment in increments:
         trajectory.append(trajectory[-1] + increment)
    return trajectory

def trajectory_to_incremets(trajectory: list[float]) -> tuple[float,list[float]]:
    """Return initial condition and list of increments that generate the grid."""
    increments = []
    for index in range(len(trajectory)-1):
        increments.append(trajectory[index + 1] - trajectory[index])
    return trajectory[0], increments
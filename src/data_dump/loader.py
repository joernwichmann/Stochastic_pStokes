import csv
from firedrake import MeshGeometry, CheckpointFile, Function
from typing import Iterable
import numpy as np

from src.discretisation.space import SpaceDiscretisation
from src.discretisation.time import TimeDiscretisation

def load_seeds(directory_name: str) -> list[int] | list[None]:
    """Return a list of stored seeds."""
    try:
        with open(directory_name + "/seeds.csv","r",newline="") as file:
            seed_reader = csv.reader(file)
            seeds = [int(seed_row[0]) for seed_row in seed_reader]
            seeds.sort()
        return seeds
    except FileNotFoundError:
        return []
    
def get_next_seed(directory_name: str) -> int:
    """Return next seed_Id."""
    try:
        return load_seeds(directory_name)[-1] + 1
    except IndexError:
        return 0
    
def get_mesh(directory_name: str, mesh_name :str) -> MeshGeometry:
    with CheckpointFile(directory_name + "/mesh.hdf5","r") as file:
        return file.load_mesh(mesh_name)
    
def get_header(directory_name: str) -> list[str]:
    with open(directory_name + "/header.csv","r") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
    return data

def get_mc_samples(directory_name: str) -> int:
    """Return number of samples."""
    return len(load_seeds(directory_name))

def get_ref_to_time_to_velocity(directory_name: str, seed_Id: int, space_disc: SpaceDiscretisation, time_disc: TimeDiscretisation) -> dict[int,dict[float,Function]]:
    return {level: _get_time_to_function(filename=directory_name + "/velocity/level_" + str(level) + "/" + str(seed_Id) + ".hdf5",
                                        function_name="velocity",
                                        time_to_id=time_disc.ref_to_time_to_id[level],
                                        mesh=space_disc.mesh)
                                        for level in time_disc.refinement_levels}

def get_ref_to_time_to_pressure(directory_name: str, seed_Id: int, space_disc: SpaceDiscretisation, time_disc: TimeDiscretisation) -> dict[int,dict[float,Function]]:
    return {level: _get_time_to_function(filename=directory_name + "/pressure/level_" + str(level) + "/" + str(seed_Id) + ".hdf5",
                                        function_name="pressure",
                                        time_to_id=time_disc.ref_to_time_to_id[level],
                                        mesh=space_disc.mesh)
                                        for level in time_disc.refinement_levels} 

def get_ref_to_noise_increments(directory_name: str, seed_Id: int, refinement_levels: list[int]) -> dict[int,np.ndarray]:
    return{level: _get_noise_increments(directory_name,seed_Id,level) for level in refinement_levels}


def get_velocity(directory_name: str, seed_Id: int, refinement_level: int, time_Id: int, space_disc: SpaceDiscretisation) -> Function:
    filename = directory_name + "/velocity/level_" + str(refinement_level) + "/" + str(seed_Id) + ".hdf5"
    with CheckpointFile(filename,"r") as file:
        velocity = file.load_function(mesh=space_disc.mesh,name="velocity",idx=time_Id)
    return velocity

def get_seed_to_velocity(directory_name: str, seeds: Iterable[int], refinement_level: int, time_Id: int, space_disc: SpaceDiscretisation) -> dict[int,Function]:
    return {seed: get_velocity(directory_name,seed,refinement_level,time_Id,space_disc) for seed in seeds}

### generic
def _get_time_to_function(filename: str, function_name: str, time_to_id: dict[float,int], mesh: MeshGeometry) -> dict[float,Function]:
    with CheckpointFile(filename,"r") as file:
        time_to_function = {time: file.load_function(mesh,function_name,time_to_id[time]) for time in time_to_id.keys()}
    return time_to_function

def _get_noise_increments(directory_name: str, seed_Id: int, refinement_level: int) -> np.ndarray:
    with open(directory_name + "/noise_increments/level_" + str(refinement_level) + "/" + str(seed_Id) + ".csv","r") as file:
        reader = csv.reader(file)
        data = [float(row[0]) for row in reader]
    return np.array(data)
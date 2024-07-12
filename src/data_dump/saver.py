from firedrake import Function, CheckpointFile, MeshGeometry
import csv
import logging
from typing import Iterable
from numpy import ndarray
import os

from src.discretisation.space import SpaceDiscretisation
from src.discretisation.time import TimeDiscretisation

def dump_sample(directory_name: str,
                seed_Id: int,
                ref_to_time_to_velocity: dict[int,dict[float,Function]],
                ref_to_time_to_pressure: dict[int,dict[float,Function]]
                ) -> None:
    """Save the sample with seed_Id into a dump directory."""
    logging.info(f"Store solution with seed_Id: \t {seed_Id}")
    for level in ref_to_time_to_velocity.keys():
        _dump_function(directory_name + "/velocity/level_" + str(level) + "/" + str(seed_Id),"velocity",ref_to_time_to_velocity[level])

    for level in ref_to_time_to_pressure.keys():
        _dump_function(directory_name + "/pressure/level_" + str(level) + "/" + str(seed_Id),"pressure",ref_to_time_to_pressure[level])

def dump_mesh(directory_name: str, mesh: MeshGeometry) -> None:
    _dump_mesh(directory_name + "/mesh",mesh)

def dump_seeds(directory_name: str, seeds: Iterable[int]) -> None:
    with open(directory_name + "/seeds.csv","a",newline="") as file:
        writer = csv.writer(file)
        writer.writerows([[seed] for seed in seeds])

def dump_noise(directory_name: str,
                seed_Id: int,
                ref_to_noise_increments: dict[int,ndarray]
                ) -> None:
    if not os.path.isdir(directory_name):
            os.makedirs(directory_name)

    for level in ref_to_noise_increments.keys():
        if not os.path.isdir(directory_name + "/noise_increments/level_" + str(level) + "/"):
            os.makedirs(directory_name + "/noise_increments/level_" + str(level) + "/")
        with open(directory_name + "/noise_increments/level_" + str(level) + "/" + str(seed_Id) + ".csv","w",newline="") as file:
            writer = csv.writer(file)
            writer.writerows([[noise_step] for noise_step in ref_to_noise_increments[level]])


def dump_header(directory_name: str,
                space_disc: SpaceDiscretisation,
                time_disc: TimeDiscretisation,
                initial_condition_name: str,
                noise_coefficient_name: str,
                model_name: str,
                algorithm_name: str) -> None:
    header = {
        "mesh": space_disc.mesh_object.name,
        "mesh resolution": space_disc.mesh_object.space_points,
        "velocity element": space_disc.velocity_discretisation.element,
        "velocity degree": space_disc.velocity_discretisation.degree,
        "pressure element": space_disc.pressure_discretisation.element,
        "pressure degree": space_disc.pressure_discretisation.degree,
        "initial condition": initial_condition_name,
        "noise coefficient": noise_coefficient_name,
        "model": model_name,
        "algorithm": algorithm_name,
        "initial time": time_disc.initial_time,
        "end time": time_disc.end_time,
        "minimal refinement": time_disc.refinement_levels[0],
        "maximal refinement": time_disc.refinement_levels[-1]
    }
    with open(directory_name + "/header.csv","w",newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header.keys())
        writer.writerow(header.values())
    


### generic functions    
def _dump_function(filename: str, function_name: str, time_to_function: dict[float,Function]) -> None:
    """Dumps the dictoniary 'time -> function' into a .hdf5 file"""
    with CheckpointFile(filename=filename + ".hdf5",mode="w") as file:
        for id, time in enumerate(time_to_function.keys()):
            file.save_function(f=time_to_function[time],idx=id,name=function_name)

def _dump_mesh(filename: str, mesh: MeshGeometry) -> None:
    """Dumps the mesh into a .hdf5 file."""
    with CheckpointFile(filename=filename + ".hdf5",mode="w") as file:
        file.save_mesh(mesh=mesh)

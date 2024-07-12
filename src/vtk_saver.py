import os
import shutil
import logging
from firedrake import File, Function

from src.database.loader import get_ref_to_time_to_velocity, get_ref_to_time_to_pressure
from src.discretisation.space import SpaceDiscretisation

#################### functions that store already loaded velocity and pressure in .vtk format 
def save_VTK_snapshot(name_outfile: str , func: Function) -> None:
    """Save function in vtk format."""
    outfile = File(name_outfile)
    outfile.write(func)

def save_solution_as_VTK(name_outfile: str, time_to_velocity: dict[float,Function], time_to_pressure: dict[float,Function]) -> None:
    """Save velocity and pressure in vtk format."""
    outfile =  File(name_outfile)
    for time in time_to_velocity.keys():
        time_to_velocity[time].rename("Velocity")
        time_to_pressure[time].rename("Pressure")
        outfile.write(time_to_velocity[time],time_to_pressure[time],time=time)

def save_function_as_VTK(name_outfile: str, name: str, time_to_function: dict[float,Function]) -> None:
    """Save function in vtk format."""
    outfile =  File(name_outfile)
    for time in time_to_function.keys():
        time_to_function[time].rename(name)
        outfile.write(time_to_function[time],time=time)

#################### functions that load and store, velocity and pressure in .vtk format 
def generate_VTK_by_seed(name_database: str, seed_Id: int, directory_name, vtk_file_name: str, space_disc: SpaceDiscretisation) -> None:
    """Save velocity and pressure with specified seed_id in vtk format."""
    ref_to_time_to_velocity = get_ref_to_time_to_velocity(name_database,seed_Id,space_disc)
    ref_to_time_to_pressure = get_ref_to_time_to_pressure(name_database,seed_Id,space_disc)

    #remove old data
    if os.path.isdir(directory_name):
            logging.debug(f"Remove data from directory: \t ./{directory_name}")
            shutil.rmtree("./" + directory_name)

    for level in ref_to_time_to_velocity.keys():
        outfile_name = directory_name + "/refinement_" + str(level) + "/" + vtk_file_name
        save_solution_as_VTK(outfile_name,ref_to_time_to_velocity[level],ref_to_time_to_pressure[level])

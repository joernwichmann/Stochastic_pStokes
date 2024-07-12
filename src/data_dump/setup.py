import os
import logging

from src.string_formatting import format_header
from src.data_dump.saver import dump_mesh, dump_header
from src.data_dump.loader import get_header, get_mesh
from src.discretisation.space import SpaceDiscretisation
from src.discretisation.time import TimeDiscretisation
from src.discretisation.velocity import VelocityDiscretisation
from src.discretisation.pressure import PressureDiscretisation
from src.discretisation.mesh import MeshObject

def _create_dir_if_not_exists(directory_name: str) -> None:
    """Create directory if non-existent."""
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)

def create_dump_structure_root(directory_name: str) -> None:
    """Create directory structure for data dump."""
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name + "/velocity/")
        os.makedirs(directory_name + "/pressure/")
    else:
        _create_dir_if_not_exists(directory_name + "/velocity/")
        _create_dir_if_not_exists(directory_name + "/pressure/")

def create_dump_structure_refinement(directory_name: str, list_of_ref: list[int]) -> None:
    """Create directory structure for refinement level data dump."""
    for level in list_of_ref:
        _create_dir_if_not_exists(directory_name + "/velocity/level_" + str(level) + "/")
        _create_dir_if_not_exists(directory_name + "/pressure/level_" + str(level) + "/")

def update_logfile(directory_name: str, name_logfile: str) -> None:
    """Checks if data has already been generated. If no data can be found, the log file is deleted."""
    if not os.path.isdir(directory_name):
        if os.path.isfile(name_logfile):
            os.remove(name_logfile)

def setup_data_dump(directory_name: str,
                      space_disc: SpaceDiscretisation,
                      time_disc: TimeDiscretisation,
                      initial_condition_name: str,
                      noise_coefficient_name: str,
                      model_name: str,
                      algorithm_name: str) -> None:
    """Checks if a storage directory is avaiable. If no directory can be found,
    create the directory and initialise header, and mesh file, and refinement dirs."""
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)
        dump_header(directory_name,space_disc,time_disc,initial_condition_name,noise_coefficient_name,model_name,algorithm_name)
        create_dump_structure_refinement(directory_name,list(time_disc.ref_to_time_grid.keys()))
        dump_mesh(directory_name,space_disc.mesh)
        msg_parameter = format_header("PARAMETER")
        msg_parameter += f"\nInitial condition:\t {initial_condition_name}"
        msg_parameter += f"\nNoise coefficient:\t {noise_coefficient_name}"
        logging.info(msg_parameter)
        logging.info(space_disc)
        logging.info(time_disc)

def construct_data_from_header(directory_name: str) -> tuple[SpaceDiscretisation,TimeDiscretisation,str,str,str,str]:
    """Constructs parameter from header file."""
    data = get_header(directory_name)

    (mesh_name, mesh_space_points,
     velocity_element, velocity_degree,
     pressure_element, pressure_degree,
     initial_condition_name, noise_coefficient_name,
     model_name, algorithm_name,
     initial_time, end_time,
     min_ref, max_ref) = data[-1]

    mesh_object = MeshObject(mesh_name, int(mesh_space_points))
    mesh_object.mesh = get_mesh(directory_name, mesh_name)
    velocity_disc = VelocityDiscretisation(mesh_object.mesh,velocity_element,int(velocity_degree))
    pressure_disc = PressureDiscretisation(mesh_object.mesh,pressure_element,int(pressure_degree))
    space_disc = SpaceDiscretisation(mesh_object,velocity_disc,pressure_disc)
    time_disc = TimeDiscretisation(float(initial_time),float(end_time),list(range(int(min_ref),int(max_ref)+1)))

    return space_disc, time_disc, initial_condition_name, noise_coefficient_name, model_name, algorithm_name


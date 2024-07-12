import sqlite3
from firedrake import Function
from copy import deepcopy
from contextlib import contextmanager

from src.discretisation.mesh import MeshObject
from src.discretisation.velocity import VelocityDiscretisation
from src.discretisation.pressure import PressureDiscretisation
from src.discretisation.space import SpaceDiscretisation
from src.discretisation.time import TimeDiscretisation

####################### context manager that opens and closes the database connection
@contextmanager
def open_db(name_database: str):
    connection = sqlite3.connect(name_database)
    try:
        cursor = connection.cursor()
        yield cursor
    finally:
        connection.commit()
        connection.close()

####################### generic function
def get_columns_of_table(name_database: str, tablename: str, *columnnames: str) -> tuple:
    """Return the columns '*columnnames' of table 'tablename' from the database 'name_database'."""
    with open_db(name_database) as cursor:

        query = "SELECT "
        for column in columnnames:
            query += column + ", "
        query = query[:-2]
        query += f" FROM {tablename}"

        cursor.execute(query)
        data = cursor.fetchall()
    return data

####################### internal functions that select various objects from the database
def _velocity_dofs(cursor) -> int:
    """Return number of velocity DOFs."""
    query = "SELECT nodal_Id FROM velocity_numbering"
    cursor.execute(query)
    return len(cursor.fetchall())

def _pressure_dofs(cursor) -> int:
    """Return number of pressure DOFs."""
    query = "SELECT nodal_Id FROM pressure_numbering"
    cursor.execute(query)
    return len(cursor.fetchall())

def _initial_time(cursor) -> float:
    """Return inital time."""
    query = "SELECT nodal_time FROM time_grid"
    cursor.execute(query)
    return cursor.fetchone()[0]

def _end_time(cursor) -> float:
    """Return end time."""
    query = "SELECT nodal_time FROM time_grid ORDER BY nodal_time DESC"
    cursor.execute(query)
    return cursor.fetchone()[0]

def _mc_samples(cursor) -> int:
    """Return number of MC samples."""
    query = "SELECT seed_Id FROM random_seeds"
    cursor.execute(query)
    return len(cursor.fetchall())

def _seeds(cursor) -> list[int]:
    """Return list of seed ids."""
    query = "SELECT seed_Id FROM random_seeds"
    cursor.execute(query)
    seeds = [seed[0] for seed in cursor.fetchall()]
    return seeds

def _refinement_levels(cursor) -> list[float]:
    """Return list of refinement levels."""
    query = "SELECT refinement_level FROM refinement"
    cursor.execute(query)
    refinement_levels = [refinement_level[0] for refinement_level in cursor.fetchall()]
    return refinement_levels

def _space_disc(cursor):
    """Return space discretisation."""
    query = "SELECT mesh_name, mesh_space_points, boundary_condition, velocity_element, velocity_degree, pressure_element, pressure_degree "
    query += "FROM space_parameter"
    cursor.execute(query)
    mesh_name, mesh_space_points, name_bc, velocity_element, velocity_degree, pressure_element, pressure_degree = cursor.fetchone() 
    mesh_object = MeshObject(mesh_name, mesh_space_points)
    velocity_disc = VelocityDiscretisation(mesh_object.mesh,velocity_element,velocity_degree)
    pressure_disc = PressureDiscretisation(mesh_object.mesh,pressure_element,pressure_degree)
    return SpaceDiscretisation(mesh_object,velocity_disc,pressure_disc,name_bc)

def _time_disc(cursor):
    """Return time discretisation."""
    return TimeDiscretisation(initial_time=_initial_time(cursor),end_time=_end_time(cursor),refinement_levels=_refinement_levels(cursor))

def _initial_condition(cursor) -> str:
    """Retrun name of initial condition."""
    query = "SELECT initial_condition FROM parameter"
    cursor.execute(query)
    return cursor.fetchone()[0]

def _noise_coefficient(cursor) -> str:
    """Return name of noise coefficient."""
    query = "SELECT noise_coefficient FROM parameter"
    cursor.execute(query)
    return cursor.fetchone()[0]


####################### internal functions that check if requested data is contained in database
def _validate_seed(seed_Id: int, cursor) -> None:
    seeds = _seeds(cursor)
    if not seed_Id in seeds:
        msg = "Seed not in database. \n"
        msg += f"Requested seed: \t {seed_Id}\n"
        msg += f"Available seed(s): \t {seeds[0]},...,{seeds[-1]}"
        raise ValueError(msg)

def _validate_refinement_level(refinement_level: int, cursor) -> None:
    refinement_levels = _refinement_levels(cursor)
    if not refinement_level in refinement_levels:
        msg = "Refinement level not in database. \n"
        msg += f"Requested refinement level: \t {refinement_level}\n"
        msg += f"Available refinement level(s): \t {refinement_levels}"
        raise ValueError(msg)


######################## load data from specific tables
def get_space_discretisation(name_database: str):
    """Return space discretisation."""
    with open_db(name_database) as cursor:
        space_disc = _space_disc(cursor)
    return space_disc

def get_time_discretisation(name_database: str):
    """Return time discretisation."""
    with open_db(name_database) as cursor:
        time_disc = _time_disc(cursor)
    return time_disc

def get_initial_condition(name_database: str) -> str:
    """Return name of initial condition."""
    with open_db(name_database) as cursor:
        return _initial_condition(cursor)
    
def get_noise_coefficient(name_database: str) -> str:
    """Return name of noise coefficient."""
    with open_db(name_database) as cursor:
        return _noise_coefficient(cursor)    

def get_initial_time(name_database: str) -> float:
    """Return initial time."""
    with open_db(name_database) as cursor:
        initial_time = _initial_time(cursor)
    return initial_time

def get_end_time(name_database: str) -> float:
    """Return end time."""
    with open_db(name_database) as cursor:
        end_time = _end_time(cursor)
    return end_time

def get_refinement_levels(name_database: str) -> list[int]:
    """Return list of refinement levels."""
    with open_db(name_database) as cursor:
        refinement_levels = _refinement_levels(cursor)
    return refinement_levels

def get_mc_samples(name_database: str) -> int:
    """Return number of MC samples."""
    with open_db(name_database) as cursor:
        mc_samples = _mc_samples(cursor)
        return mc_samples
    
def get_seeds(name_database: str) -> list[int]:
    """Return list of seed ids."""
    with open_db(name_database) as cursor:
        seeds = _seeds(cursor)
        return seeds

def get_next_seed(name_database: str) -> int:
    """Return next seed_Id."""
    if get_columns_of_table(name_database,"random_seeds","seed_Id") == []:
        return 0
    else:
        return get_columns_of_table(name_database,"random_seeds","seed_Id")[-1][0] +1
    

###################### loader for dictionaries that map time to various objects
def get_time_to_velocity(name_database: str, seed_Id: int, refinement_level: int, space_disc: SpaceDiscretisation) -> dict[float,Function]:
    """Return 'time -> velocity' dictionary for specified seed_Id and refinement_level."""
    with open_db(name_database) as cursor:
        
        _validate_seed(seed_Id,cursor)
        _validate_refinement_level(refinement_level,cursor)
        
        query = "SELECT nodal_time, velocity_nodal_Id, velocity_nodal_xvalue, velocity_nodal_yvalue "
        query += "FROM velocity "
        query += "WHERE seed_Id=? AND refinement_level=? "
        query += "ORDER BY nodal_time"

        cursor.execute(query,(seed_Id,refinement_level))

        u = Function(space_disc.velocity_space)
        time_to_velocity = dict()

        for time, id, xvalue, yvalue in cursor.fetchall():            
            u.dat.data[id,0] = xvalue
            u.dat.data[id,1] = yvalue
            if id == space_disc.velocity_dofs-1:
                time_to_velocity[time] = deepcopy(u)            
    return time_to_velocity

def get_time_to_pressure(name_database: str, seed_Id: int, refinement_level: int, space_disc: SpaceDiscretisation) -> dict[float,Function]:
    """Return 'time -> pressure' dictionary for specified seed_Id and refinement_level."""
    with open_db(name_database) as cursor:
        
        _validate_seed(seed_Id,cursor)
        _validate_refinement_level(refinement_level,cursor)

        query = "SELECT nodal_time, pressure_nodal_Id, pressure_nodal_value "
        query += "FROM pressure "
        query += "WHERE seed_Id=? AND refinement_level=? "
        query += "ORDER BY nodal_time"

        cursor.execute(query,(seed_Id,refinement_level))

        p = Function(space_disc.pressure_space)
        time_to_pressure = dict()

        for time, id, value in cursor.fetchall():            
            p.dat.data[id] = value
            if id == space_disc.pressure_dofs-1:
                time_to_pressure[time] = deepcopy(p)
    return time_to_pressure

def get_time_to_solution(name_database: str, seed_Id: int, refinement_level: int, 
                         space_disc: SpaceDiscretisation) -> dict[float,tuple[Function,Function]]:
    """Return 'time -> (velocity, pressure)' dictionary for specified seed_Id and refinement_level."""
    time_to_velocity = get_time_to_velocity(name_database,seed_Id,refinement_level,space_disc)
    time_to_pressure = get_time_to_pressure(name_database,seed_Id,refinement_level,space_disc)
    time_to_solution = dict()
    for time in time_to_velocity:
        time_to_solution[time] = (time_to_velocity[time], time_to_pressure[time])
    return time_to_solution

def get_time_to_noise(name_database: str, seed_Id: int, refinement_level: int) -> dict[float,float]:
    """Return 'time -> noise trajectory' dictionary for specified seed_Id and refinement_level."""
    with open_db(name_database) as cursor:

        _validate_seed(seed_Id,cursor)
        _validate_refinement_level(refinement_level,cursor)

        query = "SELECT nodal_time, nodal_noise "
        query += "FROM noise "
        query += "WHERE seed_Id=? AND refinement_level=? "

        cursor.execute(query,(seed_Id,refinement_level))

        time_to_noise = dict()
        for time, noise in cursor.fetchall():
            time_to_noise[time] = noise
    return time_to_noise


###################### loader for dictionaries that map refinement levels to various objects
def get_ref_to_time_to_velocity(name_database: str, seed_Id: int, space_disc: SpaceDiscretisation) -> dict[int,dict[float,Function]]: 
    """Return 'refinement level -> time -> velocity' dictionary for specified seed_Id."""
    with open_db(name_database) as cursor:
        refinement_levels = _refinement_levels(cursor)
    return {level: get_time_to_velocity(name_database,seed_Id,level,space_disc) for level in refinement_levels}

def get_ref_to_time_to_pressure(name_database: str, seed_Id: int, space_disc: SpaceDiscretisation) -> dict[int,dict[float,Function]]:
    """Return 'refinement level -> time -> pressure' dictionary for specified seed_Id."""
    with open_db(name_database) as cursor:
        refinement_levels = _refinement_levels(cursor)
    return {level: get_time_to_pressure(name_database,seed_Id,level,space_disc) for level in refinement_levels}

def get_ref_to_time_to_solution(name_database: str, seed_Id: int, 
                                space_disc: SpaceDiscretisation) -> dict[int,dict[float,tuple[Function,Function]]]:
    """Return 'refinement level -> time -> (velocity, pressure)' dictionary for specified seed_Id."""
    with open_db(name_database) as cursor:
        refinement_levels = _refinement_levels(cursor)
    return {level: get_time_to_solution(name_database,seed_Id,level,space_disc) for level in refinement_levels}

def get_ref_to_time_to_noise(name_database: str, seed_Id: int) -> dict[int,dict[float,float]]:
    """Return 'refinement level -> time -> noise trajectory' dictionary for specified seed_Id."""
    with open_db(name_database) as cursor:
        refinement_levels = _refinement_levels(cursor)
    return {level: get_time_to_noise(name_database,seed_Id,level) for level in refinement_levels}
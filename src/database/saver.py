from firedrake import Function
import logging

from src.database.loader import open_db
from src.discretisation.space import SpaceDiscretisation
from src.discretisation.time import TimeDiscretisation, increments_to_trajectory

############# generic functions for insertions into the database
def save_data_to_table(name_database: str, tablename: str, *data) -> None:
    """Save a single data tuple in specified table. """
    with open_db(name_database) as cursor:
        cursor.execute("PRAGMA foreign_keys = ON")

        query = f"INSERT INTO {tablename} VALUES("
        for _ in data:
            query += "?,"
        query = query[:-1]
        query += ")"
        cursor.execute(query, data)
        #sqliteConnection.commit()  

def save_much_data_to_table(name_database: str, tablename: str, data: list[tuple]) -> None:
    """Save a list of data tuples in specified table."""
    with open_db(name_database) as cursor:
        cursor.execute("PRAGMA foreign_keys = ON")

        query = f"INSERT INTO {tablename} VALUES("
        for _ in data[0]:
            query += "?,"
        query = query[:-1]
        query += ")"
        cursor.executemany(query, data)
        #sqliteConnection.commit()  
    
############# specific functions for insertions into the database
def save_time_to_velocity(name_database: str, seed_Id: int, refinement_level: int, velocity_dofs: int, time_to_velocity: dict[float,Function]) -> None:
    """Save 'time -> velocity' dictionary to database."""
    data = []
    for time in time_to_velocity:
        for dof_Id in range(velocity_dofs):
            data_tuple = seed_Id, refinement_level, time, dof_Id, time_to_velocity[time].dat.data[dof_Id,0], time_to_velocity[time].dat.data[dof_Id,1]
            data.append(data_tuple)
    save_much_data_to_table(name_database,"velocity",data)

def save_time_to_pressure(name_database: str, seed_Id: int, refinement_level: int, pressure_dofs: int, time_to_pressure: list) -> None:
    """Save 'time -> pressure' dictionary to databse."""
    data = []
    for time in time_to_pressure:
        for dof_Id in range(pressure_dofs):
            data_tuple = seed_Id, refinement_level, time, dof_Id, time_to_pressure[time].dat.data[dof_Id]
            data.append(data_tuple)
    save_much_data_to_table(name_database,"pressure",data)

def save_noise(name_database: str, seed_Id: int, refinement_level: int, time_grid: list[float], noise: list[float])  -> None:
    """Save noise to database."""
    data = []
    for k, time in enumerate(time_grid):
            data_tuple = seed_Id, refinement_level, time, noise[k]
            data.append(data_tuple)
    save_much_data_to_table(name_database,"noise",data)

def save_seed(name_database: str, seed_Id: int) -> None:
    """Save seed to database."""
    save_data_to_table(name_database, "random_seeds", seed_Id)

def save_data_to_database(name_database: str,
                          seed: int, 
                          time_disc: TimeDiscretisation,
                          space_disc: SpaceDiscretisation, 
                          ref_to_time_to_velocity: dict[int,dict[float,Function]], 
                          ref_to_time_to_pressure: dict[int,dict[float,Function]],
                          ref_to_noise_increments: dict[int,list[float]]):
    """Saves the data to the database."""
    logging.info(f"Store solution with seed_Id: \t {seed}")
    for level in time_disc.refinement_levels:
        save_time_to_velocity(name_database, seed, level, space_disc.velocity_dofs, ref_to_time_to_velocity[level])
        save_time_to_pressure(name_database, seed, level, space_disc.pressure_dofs, ref_to_time_to_pressure[level])
        save_noise(name_database,seed, level, time_disc.ref_to_time_grid[level],increments_to_trajectory(0,ref_to_noise_increments[level]))

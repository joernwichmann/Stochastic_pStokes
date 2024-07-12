import sqlite3
import logging
import os

from src.database.saver import save_data_to_table, save_much_data_to_table
from src.discretisation.space import SpaceDiscretisation
from src.discretisation.time import TimeDiscretisation
from src.utils import logstring_to_logger
from src.string_formatting import format_header

def create_database(name_database: str) -> None:
    """Create database."""
    with sqlite3.connect(name_database) as sqliteConnection: 
        cursor = sqliteConnection.cursor()

        cursor.execute("""CREATE TABLE IF NOT EXISTS refinement (
                refinement_level INTEGER PRIMARY KEY NOT NULL
            )""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS time_grid (
                refinement_level INTEGER NOT NULL, 
                nodal_time REAL NOT NULL, 
                FOREIGN KEY(refinement_level) REFERENCES refinement(refinement_level), 
                PRIMARY KEY(refinement_level, nodal_time) 
            )""")
        
        cursor.execute("""CREATE TABLE IF NOT EXISTS random_seeds (
                seed_Id INTEGER PRIMARY KEY 
            )""")
        
        cursor.execute("""CREATE TABLE IF NOT EXISTS velocity_numbering (
                nodal_Id INTEGER PRIMARY KEY
            )""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS pressure_numbering (
                nodal_Id INTEGER PRIMARY KEY
            )""")
        
        cursor.execute("""CREATE TABLE IF NOT EXISTS noise (
                seed_Id INTEGER NOT NULL,
                refinement_level INTEGER NOT NULL,
                nodal_time REAL NOT NULL,
                nodal_noise REAL NOT NULL,
                FOREIGN KEY(seed_Id) REFERENCES random_seeds(seed_Id),
                FOREIGN KEY(refinement_level, nodal_time) REFERENCES time_grid(refinement_level, nodal_time),
                PRIMARY KEY(seed_Id, refinement_level, nodal_time)
            )""")

        cursor.execute( """CREATE TABLE IF NOT EXISTS velocity (
                seed_Id INTEGER NOT NULL,
                refinement_level INTEGER NOT NULL,
                nodal_time REAL NOT NULL,
                velocity_nodal_Id INTEGER NOT NULL,
                velocity_nodal_xvalue REAL NOT NULL,
                velocity_nodal_yvalue REAL NOT NULL,
                FOREIGN KEY(seed_Id) REFERENCES random_seeds(seed_Id),
                FOREIGN KEY(refinement_level, nodal_time) REFERENCES time_grid(refinement_level, nodal_time),
                FOREIGN KEY(velocity_nodal_Id) REFERENCES velocity_numbering(nodal_Id),
                PRIMARY KEY(seed_Id, refinement_level, nodal_time, velocity_nodal_Id)
            )""")
        
        cursor.execute("""CREATE TABLE IF NOT EXISTS pressure (
                seed_Id INTEGER NOT NULL,
                refinement_level INTEGER NOT NULL,
                nodal_time REAL NOT NULL,
                pressure_nodal_Id INTEGER NOT NULL,
                pressure_nodal_value REAL NOT NULL,
                FOREIGN KEY(seed_Id) REFERENCES random_seeds(seed_Id),
                FOREIGN KEY(refinement_level, nodal_time) REFERENCES time_grid(refinement_level, nodal_time),
                FOREIGN KEY(pressure_nodal_Id) REFERENCES pressure_numbering(nodal_Id),
                PRIMARY KEY(seed_Id, refinement_level, nodal_time, pressure_nodal_Id)
            )""")
        
        cursor.execute("""CREATE TABLE IF NOT EXISTS space_parameter (
                mesh_name TEXT,
                mesh_space_points INTEGER,
                boundary_condition TEXT,
                velocity_element TEXT,
                velocity_degree INTEGER,
                pressure_element TEXT,
                pressure_degree INTEGER
            )""")
        
        cursor.execute("""CREATE TABLE IF NOT EXISTS parameter (
                noise_coefficient TEXT,
                initial_condition TEXT
            )""")

def initialise_indextables(name_database: str, ref_to_time_grid: dict[int,list[float]], dof_velocity: int, dof_pressure: int) -> None:
    """Initialise index tables based on CONFIGs."""
    for level in ref_to_time_grid:
        save_data_to_table(name_database,"refinement",level)
        data_tuple = [(level, time) for time in ref_to_time_grid[level]]
        save_much_data_to_table(name_database,"time_grid",data_tuple)
    data_tuple = [(id,) for id in range(dof_velocity)]
    save_much_data_to_table(name_database,"velocity_numbering",data_tuple)
    data_tuple = [(id,) for id in range(dof_pressure)]
    save_much_data_to_table(name_database,"pressure_numbering",data_tuple)

def write_space_parameter(name_database: str, space_disc) -> None:
    """Save space CONFIGs in database."""
    save_data_to_table(name_database, "space_parameter",
                        space_disc.mesh_object.name, space_disc.mesh_object.space_points, space_disc.name_bc,
                        space_disc.velocity_discretisation.element, space_disc.velocity_discretisation.degree,
                        space_disc.pressure_discretisation.element, space_disc.pressure_discretisation.degree)

def write_parameter(name_database: str, name_noise_coefficient: str, name_initial_condition: str):
    """Save noise coefficient and initial data specified in CONFIGs to database."""
    save_data_to_table(name_database, "parameter",name_noise_coefficient,name_initial_condition)


def initialise_index_and_header_tables(name_database: str,
                       time_disc: TimeDiscretisation,
                       space_disc: SpaceDiscretisation,
                       name_initial_condition: str,
                       name_noise_coefficient: str) -> None:
    """Initialise the index tables."""
    initialise_indextables(name_database,time_disc.ref_to_time_grid,space_disc.velocity_dofs,space_disc.pressure_dofs)
    write_space_parameter(name_database,space_disc)
    write_parameter(name_database,name_noise_coefficient,name_initial_condition)
    msg_parameter = format_header("PARAMETER")
    msg_parameter += f"\nInitial condition:\t {name_initial_condition}"
    msg_parameter += f"\nNoise coefficient:\t {name_noise_coefficient}"
    logging.info(msg_parameter)
    logging.info(space_disc)
    logging.info(time_disc)

def setup_database(name_database: str, name_logfile: str, log_level: str) -> bool:
    """Check if the database exists. 
    
    If DB exists, set initialisation of index tables to FALSE. 
    
    IF DB doesn't exist, create DB, create new log file, and set initialisation of index tables to TRUE.
    
    Return the status of initialisation of the index tables."""
    initialise_tables = False
    ### Check if db exists, otherwise create it and enable initialisation of index tables 
    if not os.path.isfile(name_database):
        #Remove old log file
        if os.path.isfile(name_logfile):
            os.remove(name_logfile)
        #Define logging
        logging.basicConfig(filename=name_logfile,format='%(asctime)s| \t %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', 
                        level=logstring_to_logger(log_level),force=True)
        logging.info(format_header("(RE)START"))
        #Create database
        logging.info(f"Create new database: \t {name_database}")
        create_database(name_database)
        initialise_tables = True
    else:
        logging.basicConfig(filename=name_logfile,format='%(asctime)s| \t %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', 
                        level=logstring_to_logger(log_level),force=True)
        logging.info(format_header("(RE)START"))
    return initialise_tables
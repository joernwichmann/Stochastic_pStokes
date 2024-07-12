from src.database.loader import get_columns_of_table

def _get_header(tablename: str, *columnnames: str) -> str:
    """Return header of the table."""
    header = f"Table: \t {tablename}\n"
    for column in columnnames:
        header += f"|\t {column} \t|"
    header += "\n_______________________________________________________________________________________________\n"
    return header

def _get_data_string(data: list[tuple]) -> str:
    """Return data as a formatted string."""
    msg_data = ""
    for columns in data:
        for column in columns:
            msg_data += f"|\t {column} \t|"
        msg_data += "\n"
    return msg_data

def show_columns_of_table(filename: str, tablename: str, *columnnames: str) -> None:
    """Displays specified columns of the table in the console log."""
    data = get_columns_of_table(filename, tablename, *columnnames)
    header = _get_header(tablename,*columnnames)
    data_string = _get_data_string(data)
    print(header)
    print(data_string)

def show_all_indextables(filename: str) -> None:
    """Displays all index tables to the console log."""
    show_columns_of_table(filename,"space_parameter",
                          "mesh_name","mesh_space_points",
                          "velocity_element","velocity_degree",
                          "pressure_element","pressure_degree")  
    show_columns_of_table(filename,"refinement","refinement_level")
    show_columns_of_table(filename,"time_grid","refinement_level","nodal_time")
    show_columns_of_table(filename,"velocity_numbering","nodal_Id")
    show_columns_of_table(filename,"pressure_numbering","nodal_Id")


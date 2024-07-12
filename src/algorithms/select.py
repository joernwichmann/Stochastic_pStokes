import logging
from typing import TypeAlias

from src.algorithms.stokes.parabolic import get_algorithm_by_name as get_Stokes_algorithm
from src.algorithms.stokes.parabolic import StokesAlgorithm
from src.algorithms.p_stokes.parabolic import get_algorithm_by_name as get_pStokes_algorithm
from src.algorithms.p_stokes.parabolic import pStokesAlgorithm
from src.string_formatting import format_header

### abstract structure of an algorithm
Algorithm: TypeAlias = StokesAlgorithm | pStokesAlgorithm

### converter that maps model and algorithm names to its implementation
def select_algorithm(model_name: str, algorithm_name: str) -> Algorithm:
    """Return requested algorithm for specified model."""
    msg = format_header("MODEL and ALGORITHM")
    msg += f"\nModel:\t{model_name}\nAlgorithm:\t{algorithm_name}"
    logging.info(msg)
    match model_name:
        case "Stokes":
            return get_Stokes_algorithm(algorithm_name)
        case "p-Stokes":
            return get_pStokes_algorithm(algorithm_name)
        case other:
            print(f"The model '{model_name}' is not available.")
            raise NotImplementedError
            
                

 
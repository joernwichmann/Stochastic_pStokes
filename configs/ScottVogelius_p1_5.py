"""Contains local parameter configuration."""
### Experimentname
NAME_EXPERIMENT: str = "SV15"

### Model
P_VALUE: float = 1.5

### Discretisation
VELOCITY_ELEMENT: str = "CG"    #see firedrake doc for available spaces
VELOCITY_DEGREE: int = 2       

PRESSURE_ELEMENT: str = "DG"    #see firedrake doc for available spaces
PRESSURE_DEGREE: int = 1

################               FILE/DIRECTORY NAMES               ############################
#Log
NAME_LOGFILE_GENERATE: str = f"{NAME_EXPERIMENT}.log"

#Vtk
VTK_DIRECTORY: str = f"vtk"

#Convergence
TIME_DIRECTORYNAME: str = f"convergence_results/{NAME_EXPERIMENT}"

#Stability
STABILITY_DIRECTORYNAME: str = f"stability_results/{NAME_EXPERIMENT}"

#Energy
ENERGY_DIRECTORYNAME: str = f"energy_results/{NAME_EXPERIMENT}"

#Statistics
STATISTICS_DIRECTORYNAME: str = f"statistic_results/{NAME_EXPERIMENT}"

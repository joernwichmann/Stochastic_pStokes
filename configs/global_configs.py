"""Contains global parameter configuration."""
################               GLOBAL configs               ############################
LOG_LEVEL: str = "info"  #supported levels: debug, info, warning, error, critical 

### Dump-location
DUMP_LOCATION: str = "sample_dump"

################               GENERATE configs               ############################
### Model
MODEL_NAME: str = "p-Stokes"  #see src.algorithms.select.py for available choices
KAPPA_VALUE: float = 0.1

### Algorithm
ALGORITHM_NAME: str =  "Implicit Euler mixed FEM linear multi Noise with sym Grad and approx of Averages" #see src.algorithms.select.py for available choices

### Data
INITIAL_CONDITION_NAME: str = "polynomial"    #see 'src.predefined_data' for available choices
NOISE_COEFFICIENT_NAME: str = "polynomial"  #see 'src.predefined_data' for available choices

### Discretisation
# Time
INITIAL_TIME: float = 0
END_TIME: float = 1
REFINEMENT_LEVELS: list[int] = list(range(2,10))

# Space
SPACE_RESOLUTION: str = "intermediate" ### supported str low intermediate high
MESH_NAME: str = "unit_square_non_singular"  #see 'src.discretisation.mesh' for available choices
NAME_BOUNDARY_CONDITION: str = "zero"  #see 'src.discretisation.mesh' for available choices

# Stochastic
MC_SAMPLES: int = 1000
NOISE_INCREMENTS: str = "average" # see 'src.noise' for available choices
NOISE_INTENSITY: float = 1

################               ANALYSE configs               ############################
#Convergence
TIME_CONVERGENCE: bool = True
TIME_COMPARISON_TYPE: str = "absolute"       ## "absolute" and "relative" are supported

#Stability
STABILITY_CHECK: bool = True

#Energy
ENERGY_CHECK: bool = True

#Statistics
STATISTICS_CHECK: bool = True

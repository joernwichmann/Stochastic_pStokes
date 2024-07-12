import numpy as np
from firedrake import Function

from src.string_formatting import format_header

def compare_measures(measure1: dict[tuple[tuple],int], measure2: dict[tuple[tuple],int]) -> tuple[float,float,float]:
    """Compare two empirical distributions. 
    
    Returns: - distance of the empirical distributions
     - number of hypercubes that define the empirical distribution of measure1
     - number of hypercubes that define the empirical distribution of measure2 
    """
    keys1 = set(measure1.keys())
    keys2 = set(measure2.keys())
    union_keys = keys1.union(keys2)
    keys2_notin_keys1 = union_keys.difference(keys1)
    keys1_notin_keys2 = union_keys.difference(keys2)

    ##exctend measures to common keys 
    for key in keys2_notin_keys1:
        measure1[key] = 0
    for key in keys1_notin_keys2:
        measure2[key] = 0

    compare_dinstance = 0
    for key in union_keys:
        compare_dinstance += abs(measure1[key] - measure2[key])
    return compare_dinstance/(2*sum(measure1.values())), len(keys1), len(keys2)


class MeasureOnDOFs:
    """Class that contains utilities to compute an empirical distribution."""
    def __init__(self, measure_resolution: float):
        self.measure_resolution: float = measure_resolution
        self.list_of_arrays = []

    def append_list_of_arrays(self, array):
        """Add an array to self.list_of_arrays."""
        self.list_of_arrays.append(array)

    def construct_measure(self):
        """We split the high-dim euclidean space R^d into hypercubes with side-length 'measure_resolution'.
        Each vector in R^d is projected to the vertex with coordinates [measure_resolution*(k_1,...,k_d)] of the hypercube containing it. 
        We count how many vectors of 'self.list_of_arrays' are contained in the same hypercube. 
        We store the pairing {vertex: #appearance in self.list_of_arrays} in the dictionary 'self.measure'.
        """
        self.measure = dict()
        for array in self.list_of_arrays:
            #compute location of vector in high-dim space
            array_floor = np.floor(array/self.measure_resolution)
            arr_key = tuple( tuple(arr) for arr in array_floor.tolist())
            try:
                prev_value = self.measure[arr_key]
                self.measure[arr_key] = prev_value + 1
            except KeyError:
                self.measure[arr_key] = 1


class DistributionChecker:
    """Class ..."""
    def __init__(self, measure_resolution: float, refinement_level: int):
        self.measure_resolution: float = measure_resolution
        self.refinement_level: int = refinement_level
        self.msg_measure = format_header(f"Measure resolution = {measure_resolution}")
        self.msg_measure += format_header(f"Refinement level = {refinement_level}")
        self.msg_measure += "\n|__dt__|\t|___VALUE__|\t|SPREAD_OLD|\t|SPREAD_NEW|"  

    def do_comparison(self,seed_to_velocity_old: dict[int,Function],seed_to_velocity_new: dict[int,Function], time: float):
        old_measure = MeasureOnDOFs(self.measure_resolution)
        new_measure = MeasureOnDOFs(self.measure_resolution)
        ###add data
        for seed in seed_to_velocity_new.keys():
            old_measure.append_list_of_arrays(seed_to_velocity_old[seed].dat.data[:])
            new_measure.append_list_of_arrays(seed_to_velocity_new[seed].dat.data[:])
        ###construct measure
        old_measure.construct_measure()
        new_measure.construct_measure()
        value, spread_old, spread_new = compare_measures(old_measure.measure,new_measure.measure)
        self.msg_measure += f"\n|{time:5.04f}|\t|{value:10.03e}|\t|{spread_old:10d}|\t|{spread_new:10d}|"

    def __str__(self) -> str:
        return self.msg_measure


    

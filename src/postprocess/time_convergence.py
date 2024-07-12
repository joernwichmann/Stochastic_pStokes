from firedrake import Function
import csv
import os
from functools import cached_property

from src.utils import swap_dictionary_keys
from src.math.norms.stochastic import l1_stochastic, l2_stochastic, linf_stochastic
from src.math.statistics import standard_deviation
from src.math.distances.Bochner_time import BochnerTimeDistance
from src.math.distances.space import SpaceDistance
from src.postprocess.eoc import get_ref_to_EOC
from src.postprocess.processmanager import ProcessObject

def _compare_coarse_and_fine_on_Y_X(ref_to_time_to_coarse: dict[int,dict[float,Function]],
                                    time_to_fine: dict[float,Function],
                                    Y_time_distance: BochnerTimeDistance,
                                    X_space_distance: SpaceDistance) -> dict[int,float]:
    """Compute the distance of coarse approximations and the finest approximation with respect to the 'Y' time distance and 'X' space distance. 
    
    Return 'refinement level -> error' dictionary."""
    return {level: Y_time_distance(ref_to_time_to_coarse[level],time_to_fine,X_space_distance) for level in ref_to_time_to_coarse.keys()}


def _compare_coarse_and_fine_on_Y_X_relative(ref_to_time_to_coarse: dict[int,dict[float,Function]],
                                    time_to_fine: dict[float,Function],
                                    Y_time_distance: BochnerTimeDistance,
                                    X_space_distance: SpaceDistance) -> dict[int,float]:
    """Compute the relative distance of coarse approximations and the finest approximation with respect to the 'Y' time distance and 'X' space distance. 
    Rescale error by Y-X norm of fine approximation.
    
    Return 'refinement level -> error' dictionary."""
    time_to_zero = {time: Function(time_to_fine[time].function_space()) for time in time_to_fine}

    return {level: Y_time_distance(ref_to_time_to_coarse[level],time_to_fine,X_space_distance)/Y_time_distance(time_to_fine,time_to_zero,X_space_distance) 
            for level in ref_to_time_to_coarse.keys()}


class TimeComparison(ProcessObject):
    """Class that contains tools for comparison of coarse and fine functions."""
    def __init__(self, ref_to_stepsize: dict[int,float], 
                 error_name: str, 
                 time_distance: BochnerTimeDistance,
                 space_distance: SpaceDistance,
                 comparison_type: str = "absolute") -> None:
        self.seed_to_ref_to_error = dict()
        self.seed_Id = 0
        self.ref_to_stepsize = ref_to_stepsize
        self.error_name = error_name
        self.time_distance = time_distance
        self.space_distance = space_distance
        self.comparison_type = comparison_type

    def update(self, ref_to_time_to_coarse: dict[int,dict[float,Function]], time_to_fine: dict[float,Function]) -> None:
        """Add an entry to the 'seed -> refinement level -> error' dictionary."""
        match self.comparison_type:
            case "absolute":
                self.seed_to_ref_to_error[self.seed_Id] = _compare_coarse_and_fine_on_Y_X(ref_to_time_to_coarse,time_to_fine,self.time_distance,self.space_distance)
            case "relative":
                self.seed_to_ref_to_error[self.seed_Id] = _compare_coarse_and_fine_on_Y_X_relative(ref_to_time_to_coarse,time_to_fine,self.time_distance,self.space_distance)
            case other:
                print(f"The comparison type '{self.comparison_type}' is not available.")
                raise NotImplementedError
        self.seed_Id += 1

    @cached_property
    def ref_to_seed_to_error(self):
        if len(self.seed_to_ref_to_error) == 0:
            return dict()
        return swap_dictionary_keys(self.seed_to_ref_to_error)
    
    @property
    def ref_to_error_l1(self) -> dict[int,float]:
        return {level: l1_stochastic(self.ref_to_seed_to_error[level].values()) for level in self.ref_to_seed_to_error.keys()}
    
    @property
    def ref_to_error_l2(self) -> dict[int,float]:
        return {level: l2_stochastic(self.ref_to_seed_to_error[level].values()) for level in self.ref_to_seed_to_error.keys()}
    
    @property
    def ref_to_error_linf(self) -> dict[int,float]:
        return {level: linf_stochastic(self.ref_to_seed_to_error[level].values()) for level in self.ref_to_seed_to_error.keys()}
    
    @property
    def ref_to_error_deviation(self) -> dict[int,float]:
        return {level: standard_deviation(self.ref_to_seed_to_error[level].values()) for level in self.ref_to_seed_to_error.keys()}
    
    @property
    def ref_to_EOC_l1(self) -> dict[int,float]:
        return get_ref_to_EOC(self.ref_to_error_l1,self.ref_to_stepsize)
    
    @property
    def ref_to_EOC_l2(self) -> dict[int,float]:
        return get_ref_to_EOC(self.ref_to_error_l2,self.ref_to_stepsize)
    
    @property
    def ref_to_EOC_linf(self) -> dict[int,float]:
        return get_ref_to_EOC(self.ref_to_error_linf,self.ref_to_stepsize)
    
    def save(self,name_directory) -> None:
        """Save error and EOC in file."""
        header = ["dt","L1","EOC_L1","L2","EOC_L2","Linf","EOC_Linf","Standard_Deviation"]
        data = []
        for level in list(self.ref_to_stepsize.keys())[:-1]:
            data.append([self.ref_to_stepsize[level], self.ref_to_error_l1[level],self.ref_to_EOC_l1[level],
                         self.ref_to_error_l2[level], self.ref_to_EOC_l2[level],
                         self.ref_to_error_linf[level],self.ref_to_EOC_linf[level],
                         self.ref_to_error_deviation[level]])
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)

        outfile = name_directory + "/" + self.error_name + ".csv"
        with open(outfile,"w",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)
            
    def __str__(self):
        eoc_message = f"\n\n{self.comparison_type} comparison on: \t{self.error_name}"
        eoc_message += "\n|__dt__|\t|____L1____|_EOC|\t|____L2____|_EOC|\t|___Linf___|_EOC|\t|_DEVIATION|"  
        for level in list(self.ref_to_stepsize.keys())[:-1]:
            eoc_message += f"\n|{self.ref_to_stepsize[level]:5.04f}|\t"
            eoc_message += f"|{self.ref_to_error_l1[level]:10.08f}|{self.ref_to_EOC_l1[level]:3.02f}|\t"
            eoc_message += f"|{self.ref_to_error_l2[level]:10.08f}|{self.ref_to_EOC_l2[level]:3.02f}|\t"
            eoc_message += f"|{self.ref_to_error_linf[level]:10.08f}|{self.ref_to_EOC_linf[level]:3.02f}|\t"
            eoc_message += f"|{self.ref_to_error_deviation[level]:10.08f}|"
        return eoc_message



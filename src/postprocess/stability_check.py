from firedrake import Function
import csv
import os
from functools import cached_property

from src.utils import swap_dictionary_keys
from src.math.norms.stochastic import l1_stochastic, l2_stochastic, linf_stochastic
from src.math.norms.Bochner_time import BochnerTimeNorm
from src.math.norms.space import SpaceNorm
from src.math.statistics import standard_deviation
from src.postprocess.eoc import get_ref_to_EOC
from src.postprocess.processmanager import ProcessObject

def _evaluate_norm(ref_to_time_to_function: dict[int,dict[float,Function]],
                   bochner_time_norm: BochnerTimeNorm,
                   space_norm: SpaceNorm) -> dict[int,float]:
    """Evaluate the function with respect to the Bochner time norm and space norm. 

    Return 'refinement level -> norm' dictionary."""
    norm = dict()
    for level in ref_to_time_to_function.keys():
        norm[level] = bochner_time_norm(ref_to_time_to_function[level],space_norm)
    return norm


class StabilityCheck(ProcessObject):
    """Class that contains tools for checking norm stablity."""
    def __init__(self, ref_to_stepsize: dict[int,float], 
                 norm_name: str, 
                 bochner_time_norm: BochnerTimeNorm,
                 space_norm: SpaceNorm) -> None:
        self.seed_to_ref_to_norm = dict()
        self.seed_Id = 0
        self.ref_to_stepsize = ref_to_stepsize
        self.norm_name = norm_name
        self.bochner_time_norm = bochner_time_norm
        self.space_norm = space_norm

    def update(self, ref_to_time_to_function: dict[int,dict[float,Function]]) -> None:
        """Add an entry to the 'seed -> refinement level -> norm ' dictionary."""
        self.seed_to_ref_to_norm[self.seed_Id] = _evaluate_norm(ref_to_time_to_function,self.bochner_time_norm,self.space_norm)
        self.seed_Id += 1

    @cached_property
    def ref_to_seed_to_norm(self):
        if len(self.seed_to_ref_to_norm) == 0:
            return dict()
        return swap_dictionary_keys(self.seed_to_ref_to_norm)
    
    @property
    def ref_to_norm_l1(self) -> dict[int,float]:
        return {level: l1_stochastic(self.ref_to_seed_to_norm[level].values()) for level in self.ref_to_seed_to_norm.keys()}
    
    @property
    def ref_to_norm_l2(self) -> dict[int,float]:
        return {level: l2_stochastic(self.ref_to_seed_to_norm[level].values()) for level in self.ref_to_seed_to_norm.keys()}
    
    @property
    def ref_to_norm_linf(self) -> dict[int,float]:
        return {level: linf_stochastic(self.ref_to_seed_to_norm[level].values()) for level in self.ref_to_seed_to_norm.keys()}
    
    @property
    def ref_to_norm_deviation(self) -> dict[int,float]:
        return {level: standard_deviation(self.ref_to_seed_to_norm[level].values()) for level in self.ref_to_seed_to_norm.keys()}
    
    @property
    def ref_to_EOC_l1(self) -> dict[int,float]:
        return get_ref_to_EOC(self.ref_to_norm_l1,self.ref_to_stepsize)
    
    @property
    def ref_to_EOC_l2(self) -> dict[int,float]:
        return get_ref_to_EOC(self.ref_to_norm_l2,self.ref_to_stepsize)
    
    @property
    def ref_to_EOC_linf(self) -> dict[int,float]:
        return get_ref_to_EOC(self.ref_to_norm_linf,self.ref_to_stepsize)
    
    def save(self,name_directory) -> None:
        """Save norm and EOC in file."""
        header = ["dt","L1","EOC_L1","L2","EOC_L2","Linf","EOC_Linf","Standard_Deviation"]
        data = []
        for level in self.ref_to_stepsize.keys():
            data.append([self.ref_to_stepsize[level], self.ref_to_norm_l1[level],self.ref_to_EOC_l1[level],
                         self.ref_to_norm_l2[level], self.ref_to_EOC_l2[level],
                         self.ref_to_norm_linf[level],self.ref_to_EOC_linf[level],
                         self.ref_to_norm_deviation[level]])
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)

        outfile = name_directory + "/" + self.norm_name + ".csv"
        with open(outfile,"w",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)
            
    def __str__(self):
        eoc_message = f"\n\nStability on: \t{self.norm_name}"
        eoc_message += "\n|__dt__|\t|____L1____|_EOC|\t|____L2____|_EOC|\t|___Linf___|_EOC|\t|_DEVIATION|"  
        for level in self.ref_to_stepsize.keys():
            eoc_message += f"\n|{self.ref_to_stepsize[level]:5.04f}|\t"
            eoc_message += f"|{self.ref_to_norm_l1[level]:10.08f}|{self.ref_to_EOC_l1[level]:3.02f}|\t"
            eoc_message += f"|{self.ref_to_norm_l2[level]:10.08f}|{self.ref_to_EOC_l2[level]:3.02f}|\t"
            eoc_message += f"|{self.ref_to_norm_linf[level]:10.08f}|{self.ref_to_EOC_linf[level]:3.02f}|\t"
            eoc_message += f"|{self.ref_to_norm_deviation[level]:10.08f}|"
        return eoc_message



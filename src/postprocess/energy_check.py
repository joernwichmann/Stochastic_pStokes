import csv
from firedrake import Function
import os
from functools import cached_property
from numpy import ndarray

from src.utils import swap_dictionary_keys
from src.discretisation.time import TimeDiscretisation
from src.math.norms.stochastic import l1_stochastic, l2_stochastic, linf_stochastic
from src.math.energy import Energy_function
from src.math.statistics import standard_deviation
from src.plotter import plot_ref_to_time_to_function, plot_seed_to_time_to_number, plot_seed_to_time_to_number_and_increments
from src.postprocess.processmanager import ProcessObject

def _evaluate_energy(time_to_function: dict[float,Function], energy: Energy_function) -> dict[float,float]:
    """Evaluate the energy. 

    Return 'time -> energy' dictionary."""
    return energy(time_to_function)

class Energy(ProcessObject):
    """Class that contains tools for computing the energy."""
    def __init__(self, 
                 time_disc: TimeDiscretisation,
                 energy_name: str, 
                 energy_function: Energy_function) -> None:
        self.seed_to_ref_to_time_to_energy = dict()
        self.seed_Id = 0
        self.time_disc = time_disc
        self.seed_to_ref_to_noise_increments = dict()
        self.energy_name = energy_name
        self.energy_function = energy_function

    def update(self, ref_to_time_to_function: dict[int,dict[float,Function]], ref_to_noise_increments: list[int,ndarray],) -> None:
        """Add an entry to the 'seed -> refinement level -> time -> energy ' dictionary."""
        self.seed_to_ref_to_time_to_energy[self.seed_Id] = {level: _evaluate_energy(ref_to_time_to_function[level],self.energy_function) 
                                                       for level in ref_to_time_to_function.keys()}
        self.seed_to_ref_to_noise_increments[self.seed_Id] = ref_to_noise_increments
        self.seed_Id += 1

    @cached_property
    def ref_to_seed_to_time_to_energy(self):
        if len(self.seed_to_ref_to_time_to_energy) == 0:
            return dict()
        return swap_dictionary_keys(self.seed_to_ref_to_time_to_energy)
    
    @cached_property
    def ref_to_time_to_seed_to_energy(self):
        if len(self.seed_to_ref_to_time_to_energy) == 0:
            return dict()
        return {level: swap_dictionary_keys(self.ref_to_seed_to_time_to_energy[level]) for level in self.ref_to_seed_to_time_to_energy}
    
    @cached_property
    def ref_to_seed_to_noise_increments(self):
        if len(self.seed_to_ref_to_noise_increments) == 0:
            return dict()
        return swap_dictionary_keys(self.seed_to_ref_to_noise_increments)
    
    @property
    def ref_to_time_to_energy_l1(self) -> dict[int,dict[float,float]]:
        ref_to_time_to_energy = {level: {time: l1_stochastic(self.ref_to_time_to_seed_to_energy[level][time].values()) 
                                         for time in self.time_disc.ref_to_time_grid[level]} 
                                for level in self.ref_to_seed_to_time_to_energy}
        return ref_to_time_to_energy
    
    @property
    def ref_to_time_to_energy_l2(self) -> dict[int,dict[float,float]]:
        ref_to_time_to_energy = {level: {time: l2_stochastic(self.ref_to_time_to_seed_to_energy[level][time].values())
                                         for time in self.time_disc.ref_to_time_grid[level]} 
                                for level in self.ref_to_seed_to_time_to_energy}
        return ref_to_time_to_energy
    
    @property
    def ref_to_time_to_energy_linf(self) -> dict[int,dict[float,float]]:
        ref_to_time_to_energy = {level: {time: linf_stochastic(self.ref_to_time_to_seed_to_energy[level][time].values()) 
                                         for time in self.time_disc.ref_to_time_grid[level]} 
                                for level in self.ref_to_seed_to_time_to_energy}
        return ref_to_time_to_energy
    
    @property
    def ref_to_time_to_energy_deviation(self) -> dict[int,dict[float,float]]:
        ref_to_time_to_energy = {level: {time: standard_deviation(self.ref_to_time_to_seed_to_energy[level][time].values()) 
                                         for time in self.time_disc.ref_to_time_grid[level]} 
                                for level in self.ref_to_seed_to_time_to_energy}
        return ref_to_time_to_energy
    
    def save(self, name_directory: str) -> None:
        """Save 'time -> energy' in .csv files."""
        header = ["time","L1","L2","Linf","Standard_Deviation"]
        ref_to_data = {}
        for level in self.time_disc.refinement_levels:
            ref_to_data[level] = [[time, 
                                   self.ref_to_time_to_energy_l1[level][time],
                                   self.ref_to_time_to_energy_l2[level][time],
                                   self.ref_to_time_to_energy_linf[level][time],
                                   self.ref_to_time_to_energy_deviation[level][time]] 
                                   for time in self.time_disc.ref_to_time_grid[level]]
            
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)

        new_dict_name = name_directory + "/" + self.energy_name 
        if not os.path.isdir(new_dict_name):
            os.makedirs(new_dict_name)

        for level in self.time_disc.refinement_levels:
            outfile = new_dict_name + "/refinement_" + str(level) + ".csv"
            with open(outfile,"w",newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(ref_to_data[level])

    def plot(self, name_directory: str) -> None:
        """Save 'time -> energy'."""
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)

        new_dict_name = name_directory + "/" + self.energy_name 
        if not os.path.isdir(new_dict_name):
            os.makedirs(new_dict_name)
        plot_ref_to_time_to_function({self.energy_name + "_L1": self.ref_to_time_to_energy_l1, 
                                      self.energy_name + "_Deviation": self.ref_to_time_to_energy_deviation},new_dict_name + "/L1.png",["log","log"])
        plot_ref_to_time_to_function({self.energy_name + "_L2": self.ref_to_time_to_energy_l2, 
                                      self.energy_name + "_Deviation": self.ref_to_time_to_energy_deviation},new_dict_name + "/L2.png",["log","log"])
        plot_ref_to_time_to_function({self.energy_name + "_Linf": self.ref_to_time_to_energy_linf, 
                                      self.energy_name + "_Deviation": self.ref_to_time_to_energy_deviation},new_dict_name + "/Linf.png",["log","log"])
    """       
    def plot_individual(self, name_directory: str) -> None:
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)

        new_dict_name = name_directory + "/" + self.energy_name 
        if not os.path.isdir(new_dict_name):
            os.makedirs(new_dict_name)

        for level in self.ref_to_seed_to_time_to_energy.keys():
            plot_seed_to_time_to_number(self.ref_to_seed_to_time_to_energy[level],new_dict_name + "/level_" + str(level) + ".png","level " + str(level),"log")
    """ 

    def plot_individual(self, name_directory: str) -> None:
        """Save 'time -> energy'."""
        if not os.path.isdir(name_directory):
            os.makedirs(name_directory)

        new_dict_name = name_directory + "/" + self.energy_name 
        if not os.path.isdir(new_dict_name):
            os.makedirs(new_dict_name)

        for level in self.ref_to_seed_to_time_to_energy.keys():
            plot_seed_to_time_to_number_and_increments(self.ref_to_seed_to_time_to_energy[level],
                                                       self.ref_to_seed_to_noise_increments[level],
                                                       new_dict_name + "/level_" + str(level) + ".png",
                                                       "level " + str(level),
                                                       "log",
                                                       "log")
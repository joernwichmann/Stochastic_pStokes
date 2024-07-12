import numpy as np
import os
import shutil
from firedrake import FunctionSpace, Function

from src.vtk_saver import save_function_as_VTK


def _update_mean(samples: int, old_mean_matrix: np.ndarray, update_matrix: np.ndarray) -> np.ndarray:
    """Update the mean value."""
    if not old_mean_matrix.shape == update_matrix.shape:
        msg = "Shape of mean array und update array do not match."
        msg += f"\nShape mean array:\t {old_mean_matrix.shape}"
        msg += f"\nShape update array:\t {update_matrix.shape}"
        raise ValueError(msg)
    return samples/(samples + 1)*old_mean_matrix + 1/(samples+1)*update_matrix

def _update_square(samples: int, old_square_matrix: np.ndarray, update_matrix: np.ndarray) -> np.ndarray:
    """Update the normalised second moment."""
    if not old_square_matrix.shape == update_matrix.shape:
        msg = "Shape of mean array und update array do not match."
        msg += f"\nShape mean array:\t {old_square_matrix.shape}"
        msg += f"\nShape update array:\t {update_matrix.shape}"
        raise ValueError(msg)
    return samples/(samples + 1)*old_square_matrix + 1/(samples+1)*np.power(update_matrix,2)



class StatisticsObject:
    """Class that contains utilities for the computation of mean, second moment, and deviation of 'ref -> time -> function' dictionary."""
    def __init__(self, name: str, ref_to_time_grid: dict[int,list[float]], function_space: FunctionSpace):
        self.name = name
        self.function_space = function_space
        self.ref_to_time_grid = ref_to_time_grid
        self.ref_to_time_to_function_mean = {level: {time: Function(function_space) for time in ref_to_time_grid[level]} for level in ref_to_time_grid.keys()}
        self.ref_to_time_to_function_square = {level: {time: Function(function_space) for time in ref_to_time_grid[level]} for level in ref_to_time_grid.keys()}
        self.samples = 0
        

    def update(self,ref_to_time_to_function) -> None:
        """Add a sample to mean and second moment."""
        for level in self.ref_to_time_to_function_mean.keys():
            for time in self.ref_to_time_to_function_mean[level].keys():
                #update mean
                self.ref_to_time_to_function_mean[level][time].dat.data[:] = _update_mean(self.samples,
                                                                                       self.ref_to_time_to_function_mean[level][time].dat.data,
                                                                                       ref_to_time_to_function[level][time].dat.data)
                #update square
                self.ref_to_time_to_function_square[level][time].dat.data[:] = _update_square(self.samples,
                                                                                       self.ref_to_time_to_function_square[level][time].dat.data,
                                                                                       ref_to_time_to_function[level][time].dat.data)
        self.samples += 1
    
    @property
    def ref_to_time_to_function_deviation(self) -> dict[int,dict[float,Function]]:
        """Compute the standard deviation based on mean and second moment."""
        ref_to_time_to_function_dev = {level: {time: Function(self.function_space) for time in self.ref_to_time_grid[level]} for level in self.ref_to_time_grid.keys()}
        for level in self.ref_to_time_to_function_mean.keys():
            for time in self.ref_to_time_to_function_mean[level].keys():
                #Analytically the deviation is always non-negative and there is no issue to take the square root. 
                #But numerically it might happen that the deviation is negative, leading to difficulties. We avoid this by first truncating negative values to 0.
                dev = self.ref_to_time_to_function_square[level][time].dat.data - np.power(self.ref_to_time_to_function_mean[level][time].dat.data,2)
                zeros = np.zeros(dev.shape)
                dev_zeros = np.maximum(dev,zeros)
                ref_to_time_to_function_dev[level][time].dat.data[:] = np.sqrt(dev_zeros)
        return ref_to_time_to_function_dev

    def _save_mean(self, name_directory: str) -> None:
        """Save the mean value in vtk format."""
        for level in self.ref_to_time_to_function_mean.keys():
            outfile_name = name_directory + "/refinement_" + str(level) + "/mean.pvd"
            save_function_as_VTK(outfile_name,self.name + "_mean",self.ref_to_time_to_function_mean[level])

    def _save_deviation(self, name_directory: str) -> None:
        """Save the mean value in vtk format."""
        for level in self.ref_to_time_to_function_mean.keys():
            outfile_name = name_directory + "/refinement_" + str(level) + "/deviation.pvd"
            save_function_as_VTK(outfile_name,self.name + "_dev",self.ref_to_time_to_function_deviation[level])

    def save(self,name_directory: str) -> None:
        """Save mean and standard deviation in vtk format."""
        save_directory = name_directory + "/" + self.name
        #remove old data
        if os.path.isdir(save_directory):
                shutil.rmtree("./" + save_directory)
        self._save_mean(save_directory)
        self._save_deviation(save_directory)
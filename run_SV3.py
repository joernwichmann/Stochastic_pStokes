"""Generate samples with configuration specified in 'config.py'."""
from firedrake import *
import numpy as np
import logging
from time import process_time_ns
from functools import partial

from src.discretisation.space import get_space_discretisation_from_CONFIG, SpaceDiscretisation
from src.discretisation.time import TimeDiscretisation
from src.data_dump.setup import  update_logfile
from src.algorithms.select import Algorithm, select_algorithm
from src.noise import SamplingStrategy, select_sampling
from src.predefined_data import get_function
from src.string_formatting import format_runtime, format_header
from src.utils import logstring_to_logger

from src.math.distances.space import l2_distance, h1_distance, V_distance, V_sym_distance
from src.math.distances.Bochner_time import linf_X_distance, l2_X_distance, end_time_X_distance, h_minus1_X_distance, w_minus1_inf_X_distance
from src.math.norms.space import l2_space, h1_space, hdiv_space
from src.math.norms.Bochner_time import linf_X_norm, l2_X_norm, end_time_X_norm, h_minus1_X_norm
from src.math.energy import kinetic_energy, potential_energy
from src.postprocess.time_convergence import TimeComparison
from src.postprocess.stability_check import StabilityCheck
from src.postprocess.energy_check import Energy
from src.postprocess.statistics import StatisticsObject
from src.postprocess.processmanager import ProcessManager

#load global and lokal configs
from configs import ScottVogelius_p3 as cf
from configs import global_configs as gcf

def generate_one(time_disc: TimeDiscretisation,
                 space_disc: SpaceDiscretisation,
                 initial_condition: Function,
                 noise_coefficient: Function,
                 p_value: float,
                 kappa_value: float,
                 noise_intensity: float,
                 algorithm: Algorithm,
                 sampling_strategy: SamplingStrategy) -> tuple[dict[int,list[float]], dict[int,dict[float,Function]], dict[int,dict[float,Function]]]:
    """Run the numerical experiment once. 
    
    Return noise and solution."""
    ### Generate noise on all refinement levels
    ref_to_noise_increments = sampling_strategy(time_disc.refinement_levels,time_disc.initial_time,time_disc.end_time)
    ref_to_time_to_velocity = dict()
    ref_to_time_to_pressure = dict()
    for level in ref_to_noise_increments: 
        ### Solve algebraic system
        ref_to_time_to_velocity[level], ref_to_time_to_pressure[level] = algorithm(
            space_disc=space_disc,
            time_grid=time_disc.ref_to_time_grid[level],
            noise_steps= ref_to_noise_increments[level],
            noise_coefficient=noise_coefficient,
            initial_condition=initial_condition,
            Reynolds_number=1,
            p_value=p_value,
            kappa_value=kappa_value,
            noise_intensity=noise_intensity
            )
    return ref_to_noise_increments, ref_to_time_to_velocity, ref_to_time_to_pressure


def generate() -> None:
    """Runs the experiment multiple times."""

    update_logfile(gcf.DUMP_LOCATION,cf.NAME_LOGFILE_GENERATE)

    logging.basicConfig(filename=cf.NAME_LOGFILE_GENERATE,format='%(asctime)s| \t %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', 
                        level=logstring_to_logger(gcf.LOG_LEVEL),force=True)


    # define discretisation
    space_disc = get_space_discretisation_from_CONFIG(name_mesh=gcf.MESH_NAME,
                                                      space_resolution=gcf.SPACE_RESOLUTION,
                                                      velocity_element=cf.VELOCITY_ELEMENT,
                                                      velocity_degree=cf.VELOCITY_DEGREE,
                                                      pressure_element=cf.PRESSURE_ELEMENT,
                                                      pressure_degree=cf.PRESSURE_DEGREE,
                                                      name_bc=gcf.NAME_BOUNDARY_CONDITION
                                                      )
    logging.info(space_disc)

    time_disc = TimeDiscretisation(initial_time=gcf.INITIAL_TIME, end_time=gcf.END_TIME,refinement_levels=gcf.REFINEMENT_LEVELS)
    logging.info(time_disc)

    # define data
    initial_condition = get_function(gcf.INITIAL_CONDITION_NAME,space_disc)
    noise_coefficient = get_function(gcf.NOISE_COEFFICIENT_NAME,space_disc)

    # select algorithm
    algorithm = select_algorithm(gcf.MODEL_NAME,gcf.ALGORITHM_NAME)

    #select sampling
    sampling_strategy = select_sampling(gcf.NOISE_INCREMENTS)

    #Initialise process managers to handle data processing
    if gcf.TIME_CONVERGENCE:
        time_convergence_velocity = ProcessManager([
            TimeComparison(time_disc.ref_to_time_stepsize,"Linf_L2_velocity",linf_X_distance,l2_distance,gcf.TIME_COMPARISON_TYPE),
            TimeComparison(time_disc.ref_to_time_stepsize,"L2_Vsym_velocity",l2_X_distance,partial(V_sym_distance,kappa_value=gcf.KAPPA_VALUE,p_value=cf.P_VALUE),gcf.TIME_COMPARISON_TYPE)
        ])
        time_convergence_pressure = ProcessManager([
            TimeComparison(time_disc.ref_to_time_stepsize,"L2_L2_pressure",l2_X_distance,l2_distance,gcf.TIME_COMPARISON_TYPE),
           ])

    if gcf.STABILITY_CHECK:
        stability_check_velocity = ProcessManager([
            StabilityCheck(time_disc.ref_to_time_stepsize,"L2_Hdiv_velocity",l2_X_norm,hdiv_space)
        ])
        stability_check_pressure = ProcessManager([
            StabilityCheck(time_disc.ref_to_time_stepsize,"L2_L2_pressure",l2_X_norm,l2_space),
            StabilityCheck(time_disc.ref_to_time_stepsize,"H-1_L2_pressure",h_minus1_X_norm,l2_space)
        ])

    if gcf.ENERGY_CHECK:
        energy_check_velocity = ProcessManager([
            Energy(time_disc,"kinetic_energy",kinetic_energy),
            Energy(time_disc,"potential_energy",potential_energy)
        ])        

    if gcf.STATISTICS_CHECK:
        statistics_velocity = StatisticsObject("velocity",time_disc.ref_to_time_grid,space_disc.velocity_space)
        statistics_pressure = StatisticsObject("pressure",time_disc.ref_to_time_grid,space_disc.pressure_space)

    
    runtimes = {"solving": 0,"comparison": 0, "stability": 0, "energy": 0, "statistics": 0}
    
    ### start MC iteration
    print(format_header("START MONTE CARLO ITERATION") + f"\nRequested samples:\t{gcf.MC_SAMPLES}")
    new_seeds = range(gcf.MC_SAMPLES)

    for k in new_seeds:
        ### get solution
        print(f"{k*100/len(new_seeds):4.2f}% completed")
        time_mark = process_time_ns()
        ref_to_noise_increments, ref_to_time_to_velocity, ref_to_time_to_pressure = generate_one(time_disc,space_disc,initial_condition,noise_coefficient,cf.P_VALUE,gcf.KAPPA_VALUE,gcf.NOISE_INTENSITY,algorithm,sampling_strategy)
        runtimes["solving"] += process_time_ns()-time_mark

        #update data using solution
        if gcf.TIME_CONVERGENCE:
            time_mark = process_time_ns()
            time_to_fine_velocity = ref_to_time_to_velocity[time_disc.refinement_levels[-1]]
            time_convergence_velocity.update(ref_to_time_to_velocity,time_to_fine_velocity)
            time_to_fine_pressure = ref_to_time_to_pressure[time_disc.refinement_levels[-1]]
            time_convergence_pressure.update(ref_to_time_to_pressure,time_to_fine_pressure)
            runtimes["comparison"] += process_time_ns()-time_mark

        if gcf.STABILITY_CHECK:
            time_mark = process_time_ns()
            stability_check_velocity.update(ref_to_time_to_velocity)
            stability_check_pressure.update(ref_to_time_to_pressure)
            runtimes["stability"] += process_time_ns()-time_mark

        if gcf.ENERGY_CHECK:
            time_mark = process_time_ns()
            energy_check_velocity.update(ref_to_time_to_velocity,ref_to_noise_increments)
            runtimes["energy"] += process_time_ns()-time_mark

        if gcf.STATISTICS_CHECK:
            time_mark = process_time_ns()
            statistics_velocity.update(ref_to_time_to_velocity)
            statistics_pressure.update(ref_to_time_to_pressure)
            runtimes["statistics"] += process_time_ns()-time_mark

    
    ### storing processed data 
    if gcf.TIME_CONVERGENCE:
        logging.info(format_header("TIME CONVERGENCE") + f"\nComparisons are stored in:\t {cf.TIME_DIRECTORYNAME}/")
        logging.info(time_convergence_velocity)
        logging.info(time_convergence_pressure)

        time_convergence_velocity.save(cf.TIME_DIRECTORYNAME)
        time_convergence_pressure.save(cf.TIME_DIRECTORYNAME)

    if gcf.STABILITY_CHECK:
        logging.info(format_header("STABILITY CHECK") + f"\nStability checks are stored in:\t {cf.STABILITY_DIRECTORYNAME}/")
        logging.info(stability_check_velocity)
        logging.info(stability_check_pressure)

        stability_check_velocity.save(cf.STABILITY_DIRECTORYNAME)
        stability_check_pressure.save(cf.STABILITY_DIRECTORYNAME)

    if gcf.ENERGY_CHECK:
        logging.info(format_header("ENERGY CHECK") + f"\nEnergy checks are stored in:\t {cf.ENERGY_DIRECTORYNAME}/")
        energy_check_velocity.save(cf.ENERGY_DIRECTORYNAME)
        energy_check_velocity.plot(cf.ENERGY_DIRECTORYNAME)
        energy_check_velocity.plot_individual(cf.ENERGY_DIRECTORYNAME)


    if gcf.STATISTICS_CHECK:
        logging.info(format_header("STATISTICS") + f"\nStatistics are stored in:\t {cf.VTK_DIRECTORY + '/' + cf.STATISTICS_DIRECTORYNAME}/")
        statistics_velocity.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME)
        statistics_pressure.save(cf.VTK_DIRECTORY + "/" + cf.STATISTICS_DIRECTORYNAME)

    #show runtimes
    logging.info(format_runtime(runtimes))
    print(f"Logs saved in:\t {cf.NAME_LOGFILE_GENERATE}")

if __name__ == "__main__":
    generate()
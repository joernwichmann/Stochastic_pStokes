import matplotlib.pyplot as plt
import numpy as np


def plot_ref_to_time_to_function(name_to_ref_to_time_to_number: dict[str,dict[int,dict[float,float]]],
                                 filename: str,
                                 yscale_name: list[str]) -> None:
    """Plot each dictionary entry 'name -> 'ref -> time -> function'' in a subplot where the y-axis is scaled by the entry 
    in the list 'yscale_name' and saves it."""
    plt.figure()
    for id, name in enumerate(name_to_ref_to_time_to_number.keys()):    
        plt.subplot(len(name_to_ref_to_time_to_number),1,id+1)
        for level in name_to_ref_to_time_to_number[name]:
            plt.plot(list(name_to_ref_to_time_to_number[name][level].keys()),list(name_to_ref_to_time_to_number[name][level].values()))
        plt.legend(name_to_ref_to_time_to_number[name].keys())
        plt.title(name)
        plt.yscale(yscale_name[id])
    plt.savefig(filename)

def plot_seed_to_time_to_number(seed_to_time_to_number: dict[int,dict[float,float]],
                                 filename: str,
                                 tilename: str,
                                 yscale_name: str) -> None:
    """Plot each dictionary entry 'seed -> 'time -> number ' in one figure where the y-axis is scaled 'yscale_name'. It is stored in filename."""
    plt.figure()
    for seed in seed_to_time_to_number.keys():
        plt.plot(seed_to_time_to_number[seed].keys(),seed_to_time_to_number[seed].values())
    plt.title(tilename)
    plt.yscale(yscale_name)
    plt.savefig(filename)


def plot_seed_to_time_to_number_and_increments(seed_to_time_to_number: dict[int,dict[float,float]],
                                               seed_to_noise_increments: dict[int,np.ndarray],
                                               filename: str,
                                               tilename: str,
                                               yscale1: str,
                                               yscale2: str) -> None:
    """Plot each dictionary entry 'seed -> 'time -> number ' in one figure where the y-axis is scaled 'yscale_name'. It is stored in filename."""
    plt.figure()
    plt.title(tilename)
    plt.axis("off")

    plt.subplot(2,1,1)
    plt.ylabel("energy")
    plt.xlabel("time")
    for seed in seed_to_time_to_number.keys():
        plt.plot(seed_to_time_to_number[seed].keys(),seed_to_time_to_number[seed].values())
    plt.yscale(yscale1)
    
    plt.subplot(2,1,2)
    plt.ylabel("|dW|")
    plt.xlabel("index")
    for seed in seed_to_noise_increments.keys():
        plt.plot(np.abs(seed_to_noise_increments[seed]))
    plt.yscale(yscale2)

    plt.savefig(filename)
    plt.close()
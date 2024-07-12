import numpy as np

def get_EOC_from_lists(error: list[float], stepsize: list[float]) -> list[float]:
    """Compute the experimental order of convergence for list inputs.
    
    Return list of EOC."""
    if not len(error) == len(stepsize):
        msg_error = "Error and stepsize are not of the same size.\n"
        msg_error += f"Error length: \t {len(error)}\n"
        msg_error += f"Stepsize length: \t {len(stepsize)}"
        raise ValueError(msg_error)
    eoc = [0]
    for k in range(len(error)-1):
        eoc.append(np.log(error[k+1]/error[k])/np.log(stepsize[k+1]/stepsize[k]))
    return eoc

def get_ref_to_EOC(ref_to_error: dict[int,float], ref_to_stepsize: dict[int,float]) -> dict[int,float]:
    """Compute the experimental order of convergence for dict inputs. 
    
    Return the EOC index by refinement level."""
    EOC = get_EOC_from_lists(list(ref_to_error.values()),list(ref_to_stepsize.values()))
    return {level: EOC[k] for k, level in enumerate(ref_to_error)}
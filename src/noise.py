"""Tools for generation and modification of Gaussian increments"""
import numpy as np
import logging

from typing import TypeAlias, Callable
from src.string_formatting import format_header

### abstract sampling strategy
SamplingStrategy: TypeAlias = Callable[[list[int],float,float],dict[int, np.ndarray]]

### select implementation of sampling strategy by name
def select_sampling(noise_increments: str) -> SamplingStrategy:
    """Return requested sampling strategy."""
    msg = format_header("SAMPLING STRATEGY")
    msg += f"\n\t{noise_increments}"
    logging.info(msg)
    match noise_increments:
        case "classical":
            return get_WienerIncrements_on_ref_level
        case "average":
            return get_averagedWienerIncrements_on_ref_level
        case other:
            print(f"The sampling strategy '{noise_increments}' is not available.")
            raise NotImplementedError

####################################################### Utilities #######################################
####################################################### GENERATE #######################################
#function that generates Wiener increments on a uniform time grid with size tau
def get_WienerIncrements(N: int, tau: float) -> np.ndarray:
    '''Input: time stepsize tau, Number of intervals N 
       Output: Vector of Wiener increments'''
    return np.concatenate( np.sqrt(tau)*np.random.randn(N,1), axis = 0)

#function that generates averaged increments on a uniform time grid with size tau
def get_WienerIncrementsAveraged(N: int, tau: float) -> np.ndarray:
    '''Input: time stepsize tau, Number of intervals N 
       Output: Vector of averaged Wiener increments'''
    #generate Covariance matrix
    x = np.ones( (N,1) )
    y = np.ones( (N-1,1) )
    B = np.diagflat(4*x)
    A = np.diagflat(y,-1) + np.diagflat(y,1)
    Sigma= A+B
    
    #computes its Cholesky decomposition
    C = np.linalg.cholesky(Sigma)
    
    #generates N(0,1) independent samples
    zeta = np.random.randn(N,1)
    
    #computes the correlated Gaussian random variables
    Y = np.sqrt(tau/6)*C.dot(zeta)
    Y = np.concatenate( Y, axis=0 )
    return Y
    
#joint sampling of averaged increments and classical ones
def get_JointWienerIncrements(N: int, tau: float) -> tuple[np.ndarray, np.ndarray]:
    '''Input: time stepsize tau, Number of intervals N 
       Output: Vector of Wiener increments, Vector of averaged Wiener increments'''
    dW = np.sqrt(tau)*np.random.randn(N,1)
    z = np.sqrt(tau)*np.random.randn(N,1)
    s12 = np.sqrt(12)
    adW = (dW + np.roll(dW,1))/2 + (z - np.roll(z,1))/s12
    adW[0] = dW[0]/2 + z[0]/s12
    dW = np.concatenate( dW, axis=0 )
    adW = np.concatenate( adW, axis=0 )
    return dW, adW
    
############################################################## Coarsening #################################################    
#generates coarse stochastic increments based on fine one (Number of coarse intervals needs to devide the fine ones!)
#coarse classical increments
def coarsen_WienerIncrements(dWfine: np.ndarray, Ncoarse: int) -> np.ndarray:
    '''Input: Vector of fine Wiener increments, Number of coasre intervals N 
       Output: Vector of coarse Wiener increments'''
    Nfine = np.size(dWfine)
    ratio = int(Nfine/Ncoarse)
    dWtrans = dWfine.reshape(Ncoarse,ratio)
    dWcoarse = np.sum(dWtrans, axis=1)
    return dWcoarse
        
#coarse averaged increments
def coarsen_WienerIncrementsAveraged(dWfine: np.ndarray, Ncoarse: int) -> np.ndarray:
    '''Input: Vector of fine averaged Wiener increments, Number of coasre intervals N 
       Output: Vector of coarse averaged Wiener increments'''
    Nfine = np.size(dWfine)
    ratio = int(Nfine/Ncoarse)
    dWtrans = dWfine.reshape(Ncoarse,ratio)
    w1 = np.linspace(1/ratio, 1,ratio) - 1/ratio
    w2 = np.flip( w1 + 1/ratio, axis= 0)
    dWcoarse = np.zeros((Ncoarse,1))
    d1 = np.multiply(w1,dWtrans)
    d1shift = np.roll(d1,1,axis=0)
    d2 = np.multiply(w2,dWtrans)
    dWcoarse = np.sum( d1shift + d2,axis=1)
    dWcoarse[0] = np.sum(np.multiply(w2,dWtrans[0,:]))
    #print(dWcoarse.shape)
    return dWcoarse
    
def coarsen_JointWienerIncrements(dWfine: np.ndarray, adWfine: np.ndarray, Ncoarse: int) -> tuple[np.ndarray, np.ndarray]:
    '''Input: Vector of fine Wiener increments, Vector of fine averaged Wiener increments, Number of coasre intervals N 
       Output: Vector of coarse Wiener increments, Vector of coarse averaged Wiener increments'''
    dWcoarse = coarsen_WienerIncrements(dWfine,Ncoarse)
    adWcoarse = coarsen_WienerIncrementsAveraged(adWfine,Ncoarse)
    return dWcoarse, adWcoarse
    
    

################################################### SAMPLING STRATEGIES #######################################
def get_WienerIncrements_on_ref_level(refinement_levels: list[int], initial_time: float, end_time: float) -> dict[int, np.ndarray]:
    time_steps_fine = 2**refinement_levels[-1]
    tau_fine = (end_time - initial_time)/time_steps_fine
    noise_fine = get_WienerIncrements(time_steps_fine,tau_fine)

    noise_on_ref_level = {}
    for level in refinement_levels:
        noise_on_ref_level.update({level: coarsen_WienerIncrements(noise_fine,2**level)})

    return noise_on_ref_level

def get_averagedWienerIncrements_on_ref_level(refinement_levels: list[int], initial_time: float, end_time: float) -> dict[int, np.ndarray]:
    time_steps_fine = 2**refinement_levels[-1]
    tau_fine = (end_time - initial_time)/time_steps_fine
    _, aver_noise_fine = get_JointWienerIncrements(time_steps_fine, tau_fine) 

    aver_noise_on_ref_level = {}
    for level in refinement_levels:
        aver_noise_on_ref_level.update({level: coarsen_WienerIncrementsAveraged(aver_noise_fine,2**level)})

    return aver_noise_on_ref_level


    
####################################################### Time - space Noise ###################################
def get_JointTimeSpace(N: int, tau: float, Ndof: int) -> tuple[np.ndarray, np.ndarray]:
    '''Input: time stepsize tau, Number of intervals N, Ndof of the mesh
       Output: Vector of Wiener increments, Vector of averaged Wiener increments'''
    dW = np.sqrt(tau)*np.random.randn(N,Ndof)
    z = np.sqrt(tau)*np.random.randn(N,Ndof)
    s12 = np.sqrt(12)
    adW = (dW + np.roll(dW,1,axis = 0))/2 + (z - np.roll(z,1,axis = 0))/s12
    adW[0,:] = dW[0,:]/2 + z[0,:]/s12
    return dW, adW
    
def coarsen_JointTimeSpace(dWfine: np.ndarray, adWfine: np.ndarray, Ncoarse: int) -> tuple[np.ndarray, np.ndarray]:
    _, Ndof = np.shape(dWfine)
    dWcoarse = np.zeros( (Ncoarse,Ndof) )
    adWcoarse = np.zeros( (Ncoarse,Ndof) )
    for i in range(0,Ndof):
        dWcoarse[:,i], adWcoarse[:,i] = coarsen_JointWienerIncrements(dWfine[:,i],adWfine[:,i],Ncoarse)      
    return dWcoarse, adWcoarse

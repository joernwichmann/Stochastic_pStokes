"""Defines space distances."""
from numpy import sqrt
from firedrake import assemble, inner, dx, Function, grad
from typing import Callable, TypeAlias
from src.algorithms.nonlinearities import V_tensor, V_tensor_sym

#############################           SPACE DISTANCE
#abstract concept
SpaceDistance: TypeAlias = Callable[[Function,Function],float]

#implementation
def l2_distance(function1: Function, function2: Function) -> float:
    """Compute the L2 distance of two functions."""
    return sqrt(assemble(inner(function1 - function2, function1 - function2)*dx))

def h1_distance(function1: Function, function2: Function) -> float:
    """Compute the H1 distance of two functions."""
    return l2_distance(grad(function1),grad(function2))

##### CARE: V distances do not follow abstract concept of distance as they require info of kappa and p
##### in application first use partial application (insert the value of p and kappa) to define V distance that follows abstract distanc concept
def V_distance(function1: Function, function2: Function, kappa_value: float, p_value: float) -> float:
    """Compute the gradient V distance of two functions."""
    V1 = V_tensor(grad(function1),p_value=p_value,kappa_value=kappa_value)
    V2 = V_tensor(grad(function2),p_value=p_value,kappa_value=kappa_value)
    return l2_distance(V1,V2)

def V_sym_distance(function1: Function, function2: Function, kappa_value: float, p_value: float) -> float:
    """Compute the symmetric gradient V distance of two functions."""
    V1_sym = V_tensor_sym(grad(function1),p_value=p_value,kappa_value=kappa_value)
    V2_sym = V_tensor_sym(grad(function2),p_value=p_value,kappa_value=kappa_value)
    return l2_distance(V1_sym,V2_sym)
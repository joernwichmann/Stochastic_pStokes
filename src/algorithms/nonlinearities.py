"""File that contains useful nonlinearities"""
from firedrake import *

### symmetric gradient
def epsilon(grad_u):
    return 0.5*(grad_u + grad_u.T)

### monotone operators
def S_tensor(grad_u, p_value: float = 2.0, kappa_value: float = 0.1):
    return ( kappa_value + inner(grad_u,grad_u) )**( (p_value - 2.0)/2.0 )*grad_u

def V_tensor(grad_u, p_value: float = 2.0, kappa_value: float = 0.1):
    return ( kappa_value + inner(grad_u,grad_u) )**( (p_value - 2.0)/4.0 )*grad_u

def S_tensor_sym(grad_u, p_value: float = 2.0, kappa_value: float = 0.1):
    return S_tensor(epsilon(grad_u),p_value,kappa_value)

def V_tensor_sym(grad_u, p_value: float = 2.0, kappa_value: float = 0.1):
    return V_tensor(epsilon(grad_u),p_value,kappa_value)


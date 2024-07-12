"""Defines space norms."""
from numpy import sqrt
from firedrake import assemble, inner, dx, Function, grad, div
from typing import Callable, TypeAlias

#############################           SPACE NORMS
#abstract concept
SpaceNorm: TypeAlias = Callable[[Function],float]

#implementation
def l2_space(function: Function) -> float:
    """Compute the L2 norm of a function."""
    return sqrt(assemble(inner(function,function)*dx))

def h1_space(function: Function) -> float:
    """Compute the H1 norm of a function."""
    return sqrt(assemble(inner(grad(function),grad(function))*dx))

def hdiv_space(function: Function) -> float:
    """Compute the L2 norm of the divergence of a vector field."""
    return sqrt(assemble(inner(div(function),div(function))*dx))
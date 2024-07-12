"""Defines stochastic norms."""
from numpy import sqrt, absolute
from firedrake import assemble, inner, dx, Function, grad, div
from typing import Callable, TypeAlias, Iterable

#############################           STOCHASTIC NORMS
#abstract concept
StochasticNorm: TypeAlias = Callable[[Iterable[float]],float]

#implementation
def l2_stochastic(iter_of_float: Iterable[float]) -> float:
    """Compute the l2 norm."""
    return sqrt(sum([number**2 for number in iter_of_float])/(len(iter_of_float)))

def l1_stochastic(iter_of_float: Iterable[float]) -> float:
    """Compute the l1 norm."""
    return sum([absolute(number) for number in iter_of_float])/(len(iter_of_float))

def linf_stochastic(iter_of_float: Iterable[float]) -> float:
    """Compute the linf norm."""
    return max([absolute(number) for number in iter_of_float])

"""Define statistics."""
from typing import Iterable, TypeAlias, Callable

from src.math.norms.stochastic import l2_stochastic

#############################           STATISTICS 
#abstract concept
Statistic: TypeAlias = Callable[[Iterable[float]],float]

#implementation
def mean_value(iter_of_float: Iterable[float]) -> float:
    """Compute the mean value."""
    return sum(iter_of_float)/len(iter_of_float)

def standard_deviation(iter_of_float: Iterable[float]) -> float:
    """Compute the standard deviation."""
    mean = mean_value(iter_of_float)
    recentered_list = [number - mean for number in iter_of_float]
    return l2_stochastic(recentered_list)
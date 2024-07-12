from firedrake import FunctionSpace, MeshGeometry

from src.string_formatting import format_header

class PressureDiscretisation:
    """Store pressure parameter."""
    def __init__(self, mesh: MeshGeometry, element: str, degree: int):
        self.space = FunctionSpace(mesh,element,degree)
        self.dofs = self.space.node_set.size
        self.element = element
        self.degree = degree

    def __str__(self):
        out = format_header("PRESSURE SPACE")
        out += f"\nElement: \t {self.element}"
        out += f"\nDegree: \t {self.degree}"
        out += f"\nDOFs: \t \t {self.dofs}"
        return out
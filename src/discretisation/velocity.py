from firedrake import VectorFunctionSpace, MeshGeometry

from src.string_formatting import format_header

class VelocityDiscretisation:
    """Store velocity parameter."""
    def __init__(self, mesh: MeshGeometry, element: str, degree: int):
        self.space = VectorFunctionSpace(mesh,element,degree)
        self.dofs = self.space.node_set.size
        self.element = element
        self.degree = degree     

    def __str__(self):
        out = format_header("VELOCITY SPACE")
        out += f"\nElement: \t {self.element}"
        out += f"\nDegree: \t {self.degree}"
        out += f"\nDOFs: \t \t {self.dofs}x{self.space.value_size}"
        return out
    


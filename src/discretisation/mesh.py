import netgen
from netgen.meshing import Mesh as ngMesh

from firedrake import UnitSquareMesh, COMM_WORLD, Mesh

from src.string_formatting import format_header

### converter that maps a mesh name and some resolution parameter to an implemenation
def get_mesh(name_mesh: str, space_resolution: str, comm = COMM_WORLD):
    """Return mesh based on name and resolution."""
    match name_mesh:
        case "unit square":
            return UnitSquareMesh(32,32,name = name_mesh,comm=comm)
        case "unit_square_non_singular":
            ngmesh = ngMesh()
            match space_resolution:
                case "low":
                    ngmesh.Load("src/discretisation/mesh_files/unit_square_non_singular_0.vol")
                case "intermediate":
                    ngmesh.Load("src/discretisation/mesh_files/unit_square_non_singular_1.vol")
                case "high":
                    ngmesh.Load("src/discretisation/mesh_files/unit_square_non_singular_2.vol")
                case other:
                    raise NotImplementedError
            return Mesh(ngmesh, name = name_mesh, distribution_name = name_mesh, permutation_name= name_mesh, comm=comm)
        case "unit L-shape":
            raise NotImplementedError
        case other:
            raise NotImplementedError

class MeshObject:
    """Store mesh parameter."""
    def __init__(self, name_mesh: str, space_resolution: str, comm = COMM_WORLD):
        self.name = name_mesh
        self.space_resolution = space_resolution
        self.mesh = get_mesh(name_mesh, space_resolution, comm)

    def __str__(self):
        out = format_header("MESH")
        out += f"\nName: \t \t {self.name}"
        out += f"\nSpace resolution: \t {self.space_resolution}"
        return out

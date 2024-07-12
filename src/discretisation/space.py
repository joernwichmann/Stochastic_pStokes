from firedrake import MixedVectorSpaceBasis, VectorSpaceBasis, COMM_WORLD
from typing import Optional

from src.discretisation.mesh import MeshObject
from src.discretisation.boundary_condition import get_boundary_condition
from src.discretisation.velocity import VelocityDiscretisation
from src.discretisation.pressure import PressureDiscretisation
from src.string_formatting import format_header

class SpaceDiscretisation:
    """Groups velocity, pressure, mesh, and boundary condition parameter as space parameter."""
    def __init__(self,
                 mesh_object: MeshObject,
                 velocity_discretisation: VelocityDiscretisation,
                 pressure_discretisation: PressureDiscretisation,
                 name_bc: Optional[str] = None,
                 comm = COMM_WORLD):
        self.mesh_object = mesh_object
        self.velocity_discretisation = velocity_discretisation
        self.pressure_discretisation = pressure_discretisation

        self.velocity_space = velocity_discretisation.space
        self.velocity_dofs = velocity_discretisation.dofs
     
        self.pressure_space = pressure_discretisation.space
        self.pressure_dofs = pressure_discretisation.dofs
    
        self.mixed_space = self.velocity_space*self.pressure_space
        self.total_dofs = self.velocity_dofs*self.velocity_space.value_size + self.pressure_dofs

        self.mesh = mesh_object.mesh

        self.name_bc = name_bc
        if self.name_bc:
            self.bcs_mixed, self.bcs_vel  = get_boundary_condition(self.mesh_object.name,self.name_bc,self.mixed_space,self.velocity_space)
           
        self.null = MixedVectorSpaceBasis(self.mixed_space, [self.mixed_space.sub(0), VectorSpaceBasis(constant=True,comm=comm)])


    
    def __str__(self):
        out = format_header("SPACE PARAMETER")
        out += f"\nTotal DOFs: \t {self.total_dofs}"
        out += self.mesh_object.__str__()
        if self.name_bc:
            out += f"\nBoundary condition: \t {self.name_bc}"
        out += self.velocity_discretisation.__str__()
        out += self.pressure_discretisation.__str__()
        ### maybe add physical meshsize output
        return out

def get_space_discretisation_from_CONFIG(name_mesh: str, 
                                         space_resolution: str,
                                         velocity_element: str,
                                         velocity_degree: int,
                                         pressure_element: str,
                                         pressure_degree: int,
                                         name_bc: str,
                                         comm = COMM_WORLD) -> SpaceDiscretisation:
    """Construct a space discretisation object from CONFIGS."""
    mesh_object = MeshObject(name_mesh=name_mesh,space_resolution=space_resolution,comm=comm)
    velocity_disc = VelocityDiscretisation(mesh=mesh_object.mesh,element=velocity_element,degree=velocity_degree)
    pressure_disc = PressureDiscretisation(mesh=mesh_object.mesh,element=pressure_element,degree=pressure_degree)
    return SpaceDiscretisation(mesh_object=mesh_object,velocity_discretisation=velocity_disc,pressure_discretisation=pressure_disc,name_bc=name_bc,comm=comm)


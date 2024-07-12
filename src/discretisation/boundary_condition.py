from firedrake import DirichletBC, Constant
from typing import TypeAlias

### abstract boundary condition
#The first and second entry is the BC of mixed and velocity space, respectively
BoundaryCondition: TypeAlias = tuple[list[DirichletBC],list[DirichletBC]]

### converter that maps a name of a BC to its implementation
def get_boundary_condition(name_mesh: str, name_requested_bc: str, mixed_space, velocity_space) -> BoundaryCondition:
    match name_mesh:
        case "unit square":
            return get_bc_on_unitsquare(name_requested_bc,mixed_space,velocity_space)
        case "unit_square_non_singular":
            return get_bc_on_unitsquare_non_singular(name_requested_bc,mixed_space,velocity_space)
        case "unit L-shape":
            return get_bc_on_unit_Lshape(name_requested_bc,mixed_space,velocity_space)
        case other:
            raise NotImplementedError

#implemenation
def get_bc_on_unitsquare(name_requested_bc: str, mixed_space, velocity_space) -> BoundaryCondition:
    match name_requested_bc:
        case "lid driven cavity":
            return ( 
                [DirichletBC(mixed_space.sub(0), Constant((1, 0)), (4,)), DirichletBC(mixed_space.sub(0), Constant((0, 0)), (1, 2, 3))], 
                [DirichletBC(velocity_space, Constant((1, 0)), (4,)), DirichletBC(velocity_space, Constant((0, 0)), (1, 2, 3))]
            )
        case "zero":
            return ( 
                [DirichletBC(mixed_space.sub(0), Constant((0, 0)), (1, 2, 3, 4))], 
                [DirichletBC(velocity_space, Constant((0, 0)), (1, 2, 3, 4))]
            )
        case other:
            raise NotImplementedError
        
###TODO: Change boundary markers in generation of non-singular mesh to 1,2,3,4 instead of only 1        
def get_bc_on_unitsquare_non_singular(name_requested_bc: str, mixed_space, velocity_space) -> BoundaryCondition:
    match name_requested_bc:
        case "zero":
            return ( 
                [DirichletBC(mixed_space.sub(0), Constant((0, 0)), (1))], 
                [DirichletBC(velocity_space, Constant((0, 0)), (1))]
            )
        case other:
            raise NotImplementedError

def get_bc_on_unit_Lshape(name_requested_bc: str, mixed_space, velocity_space) -> BoundaryCondition:
    match name_requested_bc:
        case "lid driven cavity":
            raise NotImplementedError
            return ( 
                [DirichletBC(mixed_space.sub(0), Constant((1, 0)), (4,)), DirichletBC(mixed_space.sub(0), Constant((0, 0)), (1, 2, 3))], 
                [DirichletBC(velocity_space, Constant((1, 0)), (4,)), DirichletBC(velocity_space, Constant((0, 0)), (1, 2, 3))]
            )
        case "zero":
            raise NotImplementedError
            return ( 
                [DirichletBC(mixed_space.sub(0), Constant((0, 0)), (1, 2, 3, 4))], 
                [DirichletBC(velocity_space, Constant((0, 0)), (1, 2, 3, 4))]
            )
        case other:
            raise NotImplementedError
        

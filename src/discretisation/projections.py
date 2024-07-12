from firedrake import *

from src.discretisation.space import SpaceDiscretisation

def Stokes_projection(vector_field: Function, space_disc: SpaceDiscretisation) -> tuple[Function,Function]:
    """Returns a discretely divergence-free velocity (the Stokes projection of 'vector_field') and a corresponding pressure."""
    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    a = ( inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) )*dx
    L = ( inner(grad(vector_field), grad(v)) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    solve(a == L, up, bcs=space_disc.bcs_mixed, nullspace=space_disc.null)

    #Mean correction
    mean_p = Constant(assemble( inner(p,1)*dx ))
    p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

    u_l2 = assemble( inner(u,u)*dx )
    u_h1 = assemble( inner(grad(u),grad(u))*dx )
    div_norm = assemble( inner(div(u),div(u))*dx )
    p_l2 = assemble( p*p*dx )
    print(f"Initial L2-norm:\t {u_l2}\nInitial H1-norm:\t{u_h1}\nInitial L2-norm divergence:\t {div_norm}\nInitial L2-norm pressure:\t{p_l2}")

    return u, p

def HL_projection(vector_field: Function, space_disc: SpaceDiscretisation) -> tuple[Function,Function]:
    """Returns a discretely divergence-free velocity (the Helmholtz--Leray projection of 'vector_field') and a corresponding pressure."""
    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    a = ( inner(u, v) - inner(p, div(v)) + inner(div(u), q) )*dx
    L = ( inner(vector_field,v) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    solve(a == L, up, nullspace=space_disc.null)
    
    #Mean correction
    mean_p = Constant(assemble( inner(p,1)*dx ))
    p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

    u_l2 = assemble( inner(u,u)*dx )
    u_h1 = assemble( inner(grad(u),grad(u))*dx )
    div_norm = assemble( inner(div(u),div(u))*dx )
    p_l2 = assemble( p*p*dx )
    print(f"Initial L2-norm:\t {u_l2}\nInitial H1-norm:\t{u_h1}\nInitial L2-norm divergence:\t {div_norm}\nInitial L2-norm pressure:\t{p_l2}")


    return u, p

def HL_projection_withBC(vector_field: Function, space_disc: SpaceDiscretisation) -> tuple[Function,Function]:
    """Returns a discretely divergence-free velocity (the Helmholtz--Leray projection of 'vector_field') and a corresponding pressure."""
    u, p = TrialFunctions(space_disc.mixed_space)
    v, q = TestFunctions(space_disc.mixed_space)

    a = ( inner(u, v) - inner(p, div(v)) + inner(div(u), q) )*dx
    L = ( inner(vector_field,v) )*dx

    up = Function(space_disc.mixed_space)
    u, p = up.subfunctions

    solve(a == L, up, bcs=space_disc.bcs_mixed,nullspace=space_disc.null)
    
    #Mean correction
    mean_p = Constant(assemble( inner(p,1)*dx ))
    p.dat.data[:] = p.dat.data - Function(space_disc.pressure_space).assign(mean_p).dat.data

    u_l2 = assemble( inner(u,u)*dx )
    u_h1 = assemble( inner(grad(u),grad(u))*dx )
    div_norm = assemble( inner(div(u),div(u))*dx )
    p_l2 = assemble( p*p*dx )
    print(f"Initial L2-norm:\t {u_l2}\nInitial H1-norm:\t{u_h1}\nInitial L2-norm divergence:\t {div_norm}\nInitial L2-norm pressure:\t{p_l2}")


    return u, p
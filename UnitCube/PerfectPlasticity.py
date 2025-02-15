#%% #!/usr/bin/env python3

Description = """
This script performs a nonlinear finite element analysis on a unit cube mesh
using the FEniCSx library. It models both elastic and plastic deformation,
implementing a return mapping algorithm for stress correction.
"""

__author__ = ['Mathieu Simon']
__date_created__ = '14-02-2025'
__license__ = 'GPL'
__version__ = '1.0'

#%% Imports

import ufl
import argparse
import numpy as np
import pandas as pd
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt
from dolfinx import mesh, fem, io
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem

#%% Define functions

def BoundaryVertices(Mesh):

    """
    Identifies and returns the vertices at the boundaries of the mesh.
    
    Parameters:
    Mesh (dolfinx.mesh.Mesh): The mesh object.
    
    Returns:
    list: A list containing the vertices at the bottom, top, north, south, east, and west boundaries.
    """
    
    Geometry = Mesh.geometry.x
    F_Bottom = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[2], Geometry[:,2].min()))
    F_Top = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[2], Geometry[:,2].max()))
    F_North = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[1], Geometry[:,1].min()))
    F_South = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[1], Geometry[:,1].max()))
    F_East = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[0], Geometry[:,0].min()))
    F_West = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[0], Geometry[:,0].max()))
    
    return [F_Bottom, F_Top, F_North, F_South, F_East, F_West]

def YieldFunction(S, Sy):

    """
    Computes the isotropic yield function for a given stress tensor.

    This function determines whether the material is in an elastic or plastic state
    by evaluating the difference between the maximum absolute component of the 
    stress tensor `S` and the yield stress `Sy`.

    Parameters:
    -----------
    S : ufl.Tensor
        The Cauchy stress tensor (3x3) represented as a UFL expression.
    Sy : float or ufl.Expr
        The yield stress threshold of the material.

    Returns:
    --------
    ufl.Expr
        The yield function value:
        - If Y(S) <= 0 → The material remains elastic.
        - If Y(S) > 0  → Plastic yielding occurs and stress correction is required.
    """
        
    Max = abs(S[0, 0])
    for i in range(3):
        for j in range(3):
            Max = ufl.max_value(Max, abs(S[i, j]))
    return Max - Sy

def NewtonBackProjection(TrialStress, Sy, MaxIter=25, Tol=1e-6):
    """
    Newton-Raphson algorithm to project stress back onto the yield surface.

    Parameters:
    S_trial : ufl.Expr - Trial stress tensor
    sigma_y : float - Yield stress limit
    max_iters : int - Maximum Newton iterations (default: 25)
    tol : float - Convergence tolerance (default: 1e-6)

    Returns:
    ufl.Expr - Corrected stress tensor after backprojection
    """
    
    # Initialize plastic multiplier Δγ as a scalar
    dGamma = 0.0  

    MaxStress = abs(TrialStress[0, 0])
    for i in range(3):
        for j in range(3):
            MaxStress = ufl.max_value(MaxStress, abs(TrialStress[i, j]))

    # Newton-Raphson iterations
    for i in range(MaxIter):

        # Compute updated stress
        S_new = TrialStress - dGamma * (TrialStress / MaxStress)

        # Compute yield function with updated stress
        MaxStressNew = abs(S_new[0, 0])
        for i in range(3):
            for j in range(3):
                MaxStressNew = ufl.max_value(MaxStressNew, abs(S_new[i, j]))
        Y_new = MaxStressNew - Sy

        # Compute residual
        Residuals = Y_new

        # Compute Newton step
        d_residual = -1  # Derivative of residual w.r.t. Δγ is -1
        dGammaNew = dGamma - Residuals / d_residual

        # Convergence check
        Error = fem.assemble_scalar(fem.form(ufl.max_value(abs(dGammaNew - dGamma), 0) * ufl.dx))
        if Error < Tol:
            break

        dGamma = dGammaNew

    # Return final corrected stress
    return TrialStress - dGamma * (TrialStress / MaxStress)

#%% Main

def Main():

    # Test definition
    IniS = 0.0                                          # Initial state (-)
    FinS = 0.2                                         # Final state/stretch (-)
    NumberSteps = 25                                   # Number of steps (-)
    DeltaStretch = round((FinS-IniS)/NumberSteps,3)    # Stretch step (-)

    # Generate mesh
    Mesh, CellTags, Classes = io.gmshio.read_from_msh('Cube.msh', comm=MPI.COMM_WORLD, rank=0, gdim=3)
    Area = 1 * 1
    Height = 1

    # Material properties
    E = 1e4  # Young's modulus (MPa)
    Nu = 0.3  # Poisson's ratio
    Mu = E / (2 * (1 + Nu))
    Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
    Sigma_y = 250.0  # Yield stress (MPa)

    # Functions space over the mesh domain
    ElementType = 'Lagrange'
    PolDegree = 1
    Ve = ufl.VectorElement(ElementType, Mesh.ufl_cell(), PolDegree)
    V = fem.FunctionSpace(Mesh, Ve)
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    # Kinematics
    d = len(u)                         # Spatial dimension
    I = ufl.variable(ufl.Identity(d))  # Identity tensor

    # Variational formulation (Linear elasticity)
    def Epsilon(u):

        """
        Computes the strain tensor for a given displacement field.
        
        Parameters:
        u (ufl.Expr): The displacement field.
        
        Returns:
        ufl.Expr: The strain tensor.
        """

        return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
    def Sigma(u):

        """
        Computes the stress tensor for a given displacement field.
        
        Parameters:
        u (ufl.Expr): The displacement field.
        
        Returns:
        ufl.Expr: The stress tensor.
        """
        
        return Lambda * ufl.nabla_div(u) * I + 2 * Mu * Epsilon(u)
    Psi = ufl.inner(Sigma(u), Epsilon(v)) * ufl.dx

    # Boundary vertices
    Vertices = BoundaryVertices(Mesh)
    Bottom, Top, North, South, East, West = Vertices
    BCs = []

    # Lock west vertices in x
    u0 = fem.Constant(Mesh, PETSc.ScalarType((0)))
    for Vertex in West:
        DOFs = fem.locate_dofs_topological(V.sub(0), 0, Vertex)
        BCs.append(fem.dirichletbc(u0, DOFs, V.sub(0)))

    # Lock south vertices in y
    u0 = fem.Constant(Mesh, PETSc.ScalarType((0)))
    for Vertex in South:
        DOFs = fem.locate_dofs_topological(V.sub(1), 0, Vertex)
        BCs.append(fem.dirichletbc(u0, DOFs, V.sub(1)))

    # Lock bottom vertices in z
    u0 = fem.Constant(Mesh, PETSc.ScalarType((0)))
    for Vertex in Bottom:
        DOFs = fem.locate_dofs_topological(V.sub(2), 0, Vertex)
        BCs.append(fem.dirichletbc(u0, DOFs, V.sub(2)))

    # Move top surface
    u1 = fem.Constant(Mesh, PETSc.ScalarType((0)))
    for Vertex in Top:
        DOFs = fem.locate_dofs_topological(V.sub(2), 0, Vertex)
        BCs.append(fem.dirichletbc(u1, DOFs, V.sub(2)))

    # Variational formulation
    Data = pd.DataFrame()
    for t in range(NumberSteps+1):

        # Update displacement
        Displacement = t * DeltaStretch * Height
        u1.value = Displacement
        
        # Solve for displacement
        Problem = NonlinearProblem(Psi, u, BCs)
        Solver = NewtonSolver(MPI.COMM_WORLD, Problem)
        Solver.solve(u)
        
        # Check yielding and apply correction if needed
        TrialStress = Sigma(u)
        Yield = YieldFunction(TrialStress, Sigma_y)
        Y_Trial = fem.assemble_scalar(fem.form(Yield * ufl.dx))
        
        if Y_Trial > 0:
            CorrectedStress = NewtonBackProjection(TrialStress, Sigma_y)
        else:
            CorrectedStress = TrialStress 

        # Compute stress
        Te = ufl.TensorElement('CG', Mesh.ufl_cell(), 1)
        T = fem.FunctionSpace(Mesh, Te)
        Expression = fem.Expression(CorrectedStress, T.element.interpolation_points())
        Stress = fem.Function(T)
        Stress.interpolate(Expression)
        Stress.name = 'Stress'

        # Evaluate stress at the center of the cube
        P33 = Stress.eval(PETSc.ScalarType((0.5,0.5,0.5)),0)[8]
        Force = P33 * Area

        Data.loc[t,'Displacement'] = Displacement
        Data.loc[t,'Simulation'] = Force

    # Plot results
    Figure, Axis = plt.subplots(1,1)
    Axis.plot([IniS, FinS], [Sigma_y/Area, Sigma_y/Area], color=(0,0,0), linestyle='--', label='Yield limit')
    Axis.plot(Data['Displacement'], Data['Simulation'], color=(1,0,0), marker='o', label='Simulation')
    Axis.set_xlabel('Displacement')
    Axis.set_ylabel('Force')
    plt.legend()
    plt.show(Figure)

    return

if __name__ == '__main__':
    
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main()
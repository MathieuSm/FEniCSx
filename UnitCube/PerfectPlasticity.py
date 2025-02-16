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

import sys
import ufl
import argparse
import numpy as np
import pandas as pd
from mpi4py import MPI
from pathlib import Path
from petsc4py import PETSc
import matplotlib.pyplot as plt
from dolfinx import mesh, fem, io
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem

sys.path.append(str(Path(__file__).parents[1]))
from Utils import Tensor


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
    Perform the plastic correction using Newton-Raphson return mapping.

    Parameters:
    - TrialStress: The trial stress tensor.
    - PlasticStrain: The previous plastic strain tensor.
    - Sy: The yield stress.
    - MaxIter: Maximum Newton iterations.
    - Tol: Convergence tolerance.

    Returns:
    - Corrected stress tensor.
    - Updated plastic strain tensor.
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

    # Return corrected stress and updated plastic strain
    return TrialStress - dGamma * (TrialStress / MaxStress)

def ContractFourthOrderSecondOrder(E4, S):
    """
    Computes the contraction of a fourth-order tensor E4 with a second-order tensor S.
    
    Parameters:
    -----------
    E4 : ufl.Tensor (3×3×3×3)
        Fourth-order compliance tensor.
    S : ufl.Tensor (3×3)
        Second-order stress tensor.
    
    Returns:
    --------
    ufl.Tensor (3×3)
        Contracted strain tensor.
    """
    return ufl.as_tensor([
        [sum(E4[i, j, k, l] * S[k, l] for k in range(3) for l in range(3)) for j in range(3)]
        for i in range(3)
    ])

#%% Main

def Main():

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

    S4 = Tensor.Isotropic(E, Nu)
    S66 = Tensor.IsoMorphism3333_66(S4)
    E66 = np.linalg.inv(S66)
    E66 = 1/2 * (E66 + E66.T)
    E4 = Tensor.IsoMorphism66_3333(E66)

    # Test definition
    Tensile = np.linspace(0.0, 0.2, 10)
    Compression = np.linspace(0.2, -Sigma_y/E, 10)
    Closing = np.linspace(-Sigma_y/E, 0.0, 5)
    Stretches = np.hstack([Tensile, Compression[1:-1], Closing])

    # Functions space over the mesh domain
    ElementType = 'Lagrange'
    PolDegree = 1
    Ve = ufl.VectorElement(ElementType, Mesh.ufl_cell(), PolDegree)
    V = fem.FunctionSpace(Mesh, Ve)
    u = fem.Function(V)
    v = ufl.TestFunction(V)

    # Plastic strain storage
    Pe = ufl.TensorElement('CG', Mesh.ufl_cell(), 1)
    P = fem.FunctionSpace(Mesh, Pe)
    Ep = fem.Function(P)

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
    def Sigma(E):

        """
        Computes the stress tensor for a given displacement field.
        
        Parameters:
        u (ufl.Expr): The displacement field.
        
        Returns:
        ufl.Expr: The stress tensor.
        """

        # Voigt notation
        E = ufl.as_vector([E[0,0], E[1,1], E[2,2], 2*E[1,2], 2*E[0,2], 2*E[0,1]])
        S = ufl.as_vector([sum(S66[i, j] * E[j] for j in range(6)) for i in range(6)])

        # Convert Voigt notation back to tensor form
        S = ufl.as_tensor([[S[0], S[5], S[4]],
                        [S[5], S[1], S[3]],
                        [S[4], S[3], S[2]]])
        return S
    
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
    for t, Stretch in enumerate(Stretches):

        # Update displacement
        Displacement = Stretch * Height
        u1.value = Displacement

        # Weak form
        EpsTrial = Epsilon(u)  # Compute strain
        TrialStress = Sigma(EpsTrial - Ep)  # Use elastic strain only
        Psi = ufl.inner(TrialStress, Epsilon(v)) * ufl.dx
        
        # Solve for displacement
        Problem = NonlinearProblem(Psi, u, BCs)
        Solver = NewtonSolver(MPI.COMM_WORLD, Problem)
        Solver.solve(u)

        # Check yielding
        Yield = YieldFunction(TrialStress, Sigma_y)
        Y_Trial = fem.assemble_scalar(fem.form(Yield * ufl.dx))
        if Y_Trial > 0:
            
            # Back-project stress onto yield surface
            CorrectedStress = NewtonBackProjection(TrialStress, Sigma_y)
        else:
            CorrectedStress = TrialStress

        # Plastic strain
        dEp = ContractFourthOrderSecondOrder(E4, TrialStress - CorrectedStress)
        Expression = fem.Expression(dEp, P.element.interpolation_points())
        dEp = fem.Function(P)
        dEp.interpolate(Expression)
        Ep.x.array[:] += dEp.x.array[:]


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
        Data.loc[t,'Plastic Strain'] = fem.assemble_scalar(fem.form(Ep[2, 2] * ufl.dx))


    # Plot results
    Figure, Axis = plt.subplots(1,2, figsize=(10,5), dpi=200)
    Axis[0].plot([min(Stretches), max(Stretches)], [Sigma_y/Area, Sigma_y/Area], color=(0,0,0), linestyle='--')
    Axis[0].plot([min(Stretches), max(Stretches)], [-Sigma_y/Area, -Sigma_y/Area], color=(0,0,0), linestyle='--', label='Yield limit')
    Axis[0].plot(Data['Displacement'], Data['Simulation'], color=(1,0,0), marker='o')
    Axis[0].set_xlabel('Displacement')
    Axis[0].set_ylabel('Force')
    Axis[0].legend()
    Axis[1].plot(Data['Displacement'], Data['Plastic Strain'], color=(0,0,1), marker='o')
    Axis[1].set_xlabel('Displacement')
    Axis[1].set_ylabel('Plastic Strain')
    plt.tight_layout()
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
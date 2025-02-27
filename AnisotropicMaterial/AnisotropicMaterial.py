#%% #!/usr/bin/env python3

Description = """
This script performs a finite element analysis of an anisotropic material using the FEniCSx library. 
It reads a mesh from a .msh file, computes the volume of the mesh, defines material constants, 
builds compliance and stiffness matrices, and sets up the function space over the mesh domain. 
"""

__author__ = ['Mathieu Simon']
__date_created__ = '30-01-2025'
__license__ = 'GPL'
__version__ = '1.0'

#%% Imports

import sys
import ufl
import gmsh
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
from mpi4py import MPI
from pathlib import Path
from petsc4py import PETSc
import matplotlib.pyplot as plt
from dolfinx import io, fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem

sys.path.append(str(Path(__file__).parents[1]))
from Utils import Time

#%% Define functions

def InitializeGMSH(Verbosity=1):

    """
    Initializes Gmsh and sets verbosity level.
    Parameters:
    Verbosity (int): Integer setting the verbosity level.
    """

    if gmsh.is_initialized():
        gmsh.clear()
    else:
        gmsh.initialize()
    gmsh.option.setNumber('General.Verbosity', Verbosity)

def ReadMesh(MeshFile, Dimensions=3):

    """
    Reads the mesh from the given file and creates a model.
    
    Parameters:
    MeshFile (pathlib.Path): The path to the mesh file.
    Dimensions (integer): Dimension of the mesh.
    
    Returns:
    tuple: The mesh, tags, and classes from the model.
    """

    gmsh.merge(str(MeshFile))
    return io.gmshio.model_to_mesh(gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=Dimensions)

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

def KUBCs(E_Hom, Faces, Geometry, Mesh, V):

    """
    Applies kinematic uniform boundary conditions (KUBCs) to the mesh.
    
    Parameters:
    E_Hom (numpy.ndarray): The homogenized strain tensor.
    Vertices (list): List of vertices at the boundaries.
    Geometry (numpy.ndarray): The geometry of the mesh.
    Mesh (dolfinx.mesh.Mesh): The mesh object.
    V (dolfinx.fem.FunctionSpace): The function space.
    
    Returns:
    list: A list of Dirichlet boundary conditions.
    
    Reference:
    Pahr, D.H., Zysset, P.K.
    Influence of boundary conditions on computed apparent elastic properties of cancellous bone.
    Biomech Model Mechanobiol 7, 463–476 (2008).
    https://doi.org/10.1007/s10237-007-0109-7
    """


    # Reference nodes and face vertices
    V_Bottom, V_Top, V_North, V_South, V_East, V_West = Faces

    BCs = []
    Constrained = []
    for Vertice in V_West:

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(len(Loc)), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for Vertice in V_South:

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(len(Loc)), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for Vertice in V_Bottom:

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(len(Loc)), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for Vertice in V_East:

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(len(Loc)), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for Vertice in V_North:

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(len(Loc)), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    for Vertice in V_Top:

        if Vertice in Constrained:
            pass
        else:
            # Compute node position
            Loc = Geometry[Vertice]
            NewLoc = np.dot(E_Hom + np.eye(len(Loc)), Loc)

            # Displacement
            u1 = fem.Constant(Mesh,(NewLoc - Loc))

            # Apply boundary conditions and store
            DOFs = fem.locate_dofs_topological(V, 0, Vertice)
            BC = fem.dirichletbc(u1, DOFs, V)
            BCs.append(BC)
            Constrained.append(Vertice)

    return BCs

#%% Main

def Main():

    # Define paths
    MeshPath = Path(__file__).parent / 'Meshes'
    OutputPath = Path(__file__).parent / 'FEniCS'

    # List meshes
    Meshes = sorted([F for F in MeshPath.iterdir() if F.name.endswith('.msh')])

    # Define Material Constants
    E1   = 12442.5
    E2   = 12442.5
    E3   = 23770.8
    Nu23 = 0.245987
    Nu13 = 0.245987
    Nu12 = 0.34
    Mu23 = 6417.14
    Mu31 = 6417.14
    Mu12 = 4642.74

    E1   = 1e4
    E2   = 1e4
    E3   = 1e4
    Nu23 = 0.3
    Nu13 = 0.3
    Nu12 = 0.3
    Mu23 = E1 / (2*(1+Nu23))
    Mu31 = E2 / (2*(1+(Nu13/E1*E3)))
    Mu12 = E3 / (2*(1+Nu12))

    # Build compliance and stiffness matrices
    E = np.array([[1/E1, -Nu12/E1, -Nu13/E1, 0, 0, 0],
                [-Nu12/E1, 1/E2, -Nu23/E2, 0, 0, 0],
                [-Nu13/E1, -Nu23/E2, 1/E3, 0, 0, 0],
                [0, 0, 0, 1/Mu23, 0, 0],
                [0, 0, 0, 0, 1/Mu31, 0],
                [0, 0, 0, 0, 0, 1/Mu12]])

    C = np.linalg.inv(E)

    # Iterate over each ROI
    for m, MeshFile in enumerate(Meshes):
        
        # Print time
        Time.Process(1,MeshFile.name[:-4])

        # Read Mesh and create model
        InitializeGMSH()
        Mesh, CellTags, Classes = ReadMesh(MeshFile)

        # Record time
        Time.Process(1, MeshFile.name[:-4])

        # Compute volume
        Geometry = Mesh.geometry.x
        L1 = max(Geometry[:,0]) - min(Geometry[:,0])
        L2 = max(Geometry[:,1]) - min(Geometry[:,1])
        L3 = max(Geometry[:,2]) - min(Geometry[:,2])
        Volume = L1 * L2 * L3

        # Functions space over the mesh domain
        ElementType = 'Lagrange'
        PolDegree = 1
        Ve = ufl.VectorElement(ElementType, Mesh.ufl_cell(), PolDegree)
        V = fem.FunctionSpace(Mesh, Ve)
        u = ufl.TrialFunction(V)     # Incremental displacement
        v = ufl.TestFunction(V)      # Test function

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

            E = Epsilon(u)

            # Voigt notation
            E = ufl.as_vector([E[0,0], E[1,1], E[2,2], 2*E[1,2], 2*E[0,2], 2*E[0,1]])
            S = ufl.as_vector([sum(C[i, j] * E[j] for j in range(6)) for i in range(6)])

            # Convert Voigt notation back to tensor form
            S = ufl.as_tensor([[S[0], S[5], S[4]],
                            [S[5], S[1], S[3]],
                            [S[4], S[3], S[2]]])
            return S

        Psi = ufl.inner(Sigma(u), Epsilon(v)) * ufl.dx

        # Load cases
        LCs = ['Tensile1', 'Tensile2', 'Tensile3', 'Shear23', 'Shear31', 'Shear12']

        # Corresponding homogenized strain
        Value = 0.001
        E_Homs = np.zeros((6,3,3))
        E_Homs[0,0,0] = Value
        E_Homs[1,1,1] = Value
        E_Homs[2,2,2] = Value
        E_Homs[3,1,2] = Value
        E_Homs[3,2,1] = Value
        E_Homs[4,0,2] = Value
        E_Homs[4,2,0] = Value
        E_Homs[5,0,1] = Value
        E_Homs[5,1,0] = Value

        # Locate faces vertices
        Vertices = BoundaryVertices(Mesh)

        # Boundary conditions (external loads)
        f = fem.Constant(Mesh,(0.0, 0.0, 0.0))
        Load = ufl.dot(f, u) * ufl.ds

        # Solve for all loadcases
        S = np.zeros((6,6)) # Stiffness matrix
        for LoadCase, LC in enumerate(LCs):

            Time.Update((LoadCase+1)/6, LC)

            E_Hom = E_Homs[LoadCase]
            BCs = KUBCs(E_Hom, Vertices, Geometry, Mesh, V)

            # Solve problem
            Problem = LinearProblem(Psi, Load, BCs, petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})
            uh = Problem.solve()
            uh.name = 'Deformation'

            # Compute homogenized stress
            S_Matrix = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    S_Matrix[i,j] = fem.assemble_scalar(fem.form(Sigma(uh)[i,j]*ufl.dx))
            S_Hom = S_Matrix / Volume

            # Build stiffness matrix
            epsilon = [E_Hom[0,0], E_Hom[1,1], E_Hom[2,2], 2*E_Hom[1,2], 2*E_Hom[2,0], 2*E_Hom[0,1]]
            sigma = [S_Hom[0,0], S_Hom[1,1], S_Hom[2,2], S_Hom[1,2], S_Hom[2,0], S_Hom[0,1]]

            for i in range(6):
                S[i,LoadCase] = sigma[i] / epsilon[LoadCase]

        # Save stiffness matrix
        FileName = OutputPath / MeshFile.name[:-4]
        np.save(str(FileName) + '.npy', S)
        Time.Process(0,f'Sim. {m+1}/{len(Meshes)} done')

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

        
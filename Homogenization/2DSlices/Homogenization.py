#%% #!/usr/bin/env python3

Description = """
Script used to perform square homogenization using FEniCSx
doi: 10.1007/s10237-007-0109-7
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
__date__ = '12-04-2024'
__license__ = 'GPL'
__version__ = '1.0'

#%% Imports

import sys
import ufl
import gmsh
import argparse
import numpy as np
import pyvista as pv
from mpi4py import MPI
from pathlib import Path
from petsc4py import PETSc
from dolfinx import io, fem, mesh
from dolfinx.fem.petsc import LinearProblem

sys.path.append(str(Path(__file__).parents[2]))
from Utils import Time

#%% Define functions

def InitializzeGMSH(Verbosity=1):

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

def ReadMesh(MeshFile):

    """
    Reads the mesh from the given file and creates a model.
    
    Parameters:
    MeshFile (pathlib.Path): The path to the mesh file.
    
    Returns:
    tuple: The mesh, tags, and classes from the model.
    """

    gmsh.merge(str(MeshFile))
    return io.gmshio.model_to_mesh(gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=2)

def MaterialConstants(Tag=[], YoungsModulus=[], PoissonRatio=[]):
    """
    Defines material constants for the simulation.
    
    Returns:
    list: A list of tuples containing tag, Young's modulus, and Poisson's ratio.
    """
    return [(T, PETSc.ScalarType(E), PETSc.ScalarType(Nu))
    for T, E, Nu in zip(Tag, YoungsModulus, PoissonRatio)]

def compute_lame_parameters(mesh, tags, material_constants):
    """
    Computes the Lamé parameters as a function of cell tags.
    
    Parameters:
    mesh (dolfinx.mesh.Mesh): The mesh object.
    tags (dolfinx.mesh.MeshTags): The mesh tags.
    material_constants (list): A list of material constants.
    
    Returns:
    tuple: The Lamé parameters lambda and mu.
    """
    lambda_ = fem.Function(fem.FunctionSpace(mesh, ('DG', 0)))
    mu = fem.Function(fem.FunctionSpace(mesh, ('DG', 0)))
    
    for tag, E, nu in material_constants:
        cells = np.where(tags.values == tag)[0]
        lambda_.x.array[cells] = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu.x.array[cells] = E / (2 * (1 + nu))
    
    return lambda_, mu

def BoundariesVertices(Mesh):
    Geometry = Mesh.geometry.x[:,:-1]
    F_North = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[1], Geometry[:,1].min()))
    F_South = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[1], Geometry[:,1].max()))
    F_East = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[0], Geometry[:,0].min()))
    F_West = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[0], Geometry[:,0].max()))
    return [F_North, F_South, F_East, F_West]

def KUBCs(E_Hom, Vertices, Geometry, Mesh, V):

    # Reference nodes and face vertices
    V_North, V_South, V_East, V_West = Vertices

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

    return BCs

#%% Main

def Main():

    # Define paths
    MeshPath = Path(__file__).parent / 'Meshes'
    OutputPath = Path(__file__).parent / 'Simulations'

    # List meshes
    Meshes = sorted([F for F in MeshPath.iterdir() if F.name.endswith('.msh')])

    # Iterate over each ROI
    for m, MeshFile in enumerate(Meshes):

        # Print time
        Time.Process(1,MeshFile.name[:-4])

        # Read Mesh and create model
        if gmsh.is_initialized():
            gmsh.clear()
        else:
            gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 1)
        gmsh.merge(str(MeshFile))
        Mesh, Tags, Classes = io.gmshio.model_to_mesh(gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=2)

        # Define Material Constants
        E1      = PETSc.ScalarType(1E4)        # Young's modulus (Pa)
        Nu1     = PETSc.ScalarType(0.3)        # Poisson's ratio (-)
        E2      = PETSc.ScalarType(5E3)        # Young's modulus (Pa)
        Nu2     = PETSc.ScalarType(0.3)        # Poisson's ratio (-)
        E3      = PETSc.ScalarType(2E4)        # Young's modulus (Pa)
        Nu3     = PETSc.ScalarType(0.3)        # Poisson's ratio (-)

        # Define lamé parameters as function of cell tags
        Lambda = fem.Function(fem.FunctionSpace(Mesh, ('DG', 0))) # First Lamé parameter (kPa)
        Mu     = fem.Function(fem.FunctionSpace(Mesh, ('DG', 0))) # Shear modulus (kPa)
        
        for Tag, E, Nu in [[1, E1, Nu1], [2, E2, Nu2], [3, E3, Nu3]]:
            Cells = np.where(Tags.values == Tag)[0]
            Lambda.x.array[Cells] = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
            Mu.x.array[Cells] = E / (2 * (1 + Nu))

        # Stiffness matrix initialization
        S = np.zeros((3,3))

        # External surfaces
        Geometry = Mesh.geometry.x[:,:-1]
        L1 = max(Geometry[:,0]) - min(Geometry[:,0])
        L2 = max(Geometry[:,1]) - min(Geometry[:,1])
        Surface = L1 * L2

        # Functions space over the mesh domain
        ElementType = 'Lagrange'
        PolDegree = 1
        Ve = ufl.VectorElement(ElementType, Mesh.ufl_cell(), PolDegree, 2)
        V = fem.FunctionSpace(Mesh, Ve)
        u = ufl.TrialFunction(V)     # Incremental displacement
        v = ufl.TestFunction(V)      # Test function

        # Kinematics
        d = len(u)                         # Spatial dimension
        I = ufl.variable(ufl.Identity(d))  # Identity tensor

        # Variational formulation (Linear elasticity)
        def Epsilon(u):
            return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        def Sigma(u):
            return Lambda * ufl.nabla_div(u) * I + 2 * Mu * Epsilon(u)
        Psi = ufl.inner(Sigma(u), Epsilon(v)) * ufl.dx

        # Load cases
        LCs = ['Tensile1', 'Tensile2', 'Shear12']

        # Corresponding homogenized strain
        Value = 0.001
        E_Homs = np.zeros((3,2,2))
        E_Homs[0,0,0] = Value
        E_Homs[1,1,1] = Value
        E_Homs[2,0,1] = Value / 2
        E_Homs[2,1,0] = Value / 2

        # Locate vertices at the boundaries
        Vertices = BoundariesVertices(Mesh)

        # Boundary conditions (external loads)
        f = fem.Constant(Mesh, PETSc.ScalarType((0.0, 0.0)))
        Load = ufl.dot(f, u) * ufl.ds

        # Solve for all loadcases
        FileName = OutputPath / MeshFile.name[:-4]
        for LoadCase in range(3):

            Time.Update(LoadCase/3, LCs[LoadCase])

            # Define homogeneous deformation
            E_Hom = E_Homs[LoadCase]

            # Define boundary conditions
            BCs = KUBCs(E_Hom, Vertices, Geometry, Mesh, V)

            # Solve problem
            Problem = LinearProblem(Psi, Load, BCs, petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})
            uh = Problem.solve()

            # Compute homogenized stress
            S_Matrix = np.zeros((2,2))
            for i in range(2):
                for j in range(2):
                    S_Matrix[i,j] = fem.assemble_scalar(fem.form(Sigma(uh)[i,j]*ufl.dx))
            S_Hom = S_Matrix / Surface

            # Build stiffness matrix
            epsilon = [E_Hom[0,0], E_Hom[1,1], 2*E_Hom[0,1]]
            sigma = [S_Hom[0,0], S_Hom[1,1], S_Hom[0,1]]

            for i in range(3):
                S[i,LoadCase] = sigma[i] / epsilon[LoadCase]

        # Save stiffness matrix
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

        
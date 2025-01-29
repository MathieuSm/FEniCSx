#%% #!/usr/bin/env python3

Description = """
Script used to perform square homogenization using FEniCSx
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2023'
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

def MaterialConstants(Tag=[], YoungsModulus=[], PoissonRatio=[]):

    """
    Defines material constants for the simulation.
    
    Parameters:
    Tag (list): List of tags identifying different materials.
    YoungsModulus (list): List of Young's modulus values corresponding to each tag.
    PoissonRatio (list): List of Poisson's ratio values corresponding to each tag.
    
    Returns:
    list: A list of tuples, each containing a tag, Young's modulus, and Poisson's ratio.
    """

    return [(T, PETSc.ScalarType(E), PETSc.ScalarType(Nu))
    for T, E, Nu in zip(Tag, YoungsModulus, PoissonRatio)]

def LameParameters(Mesh, CellTags, MaterialConstants):

    """
    Computes the Lamé parameters as a function of cell tags.
    
    Parameters:
    mesh (dolfinx.mesh.Mesh): The mesh object.
    tags (dolfinx.mesh.MeshTags): The mesh tags.
    material_constants (list): A list of material constants.
    
    Returns:
    tuple: The Lamé parameters lambda and mu.
    """

    Lambda = fem.Function(fem.FunctionSpace(Mesh, ('DG', 0)))
    Mu = fem.Function(fem.FunctionSpace(Mesh, ('DG', 0)))
    
    for Tag, E, nu in MaterialConstants:
        Cells = np.where(CellTags.values == Tag)[0]
        Lambda.x.array[Cells] = E * nu / ((1 + nu) * (1 - 2 * nu))
        Mu.x.array[Cells] = E / (2 * (1 + nu))
    
    return Lambda, Mu

def BoundariesVertices(Mesh):

    """
    Identifies and returns the vertices at the boundaries of the mesh.
    
    Parameters:
    Mesh (dolfinx.mesh.Mesh): The mesh object.
    
    Returns:
    list: A list containing the vertices at the north, south, east, and west boundaries.
    """

    Geometry = Mesh.geometry.x[:,:-1]
    F_North = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[1], Geometry[:,1].min()))
    F_South = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[1], Geometry[:,1].max()))
    F_East = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[0], Geometry[:,0].min()))
    F_West = mesh.locate_entities_boundary(Mesh, 0, lambda x: np.isclose(x[0], Geometry[:,0].max()))

    return [F_North, F_South, F_East, F_West]

def KUBCs(E_Hom, Vertices, Geometry, Mesh, V):

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
        InitializeGMSH()
        Mesh, CellTags, Classes = ReadMesh(MeshFile, 2)

        # Define Material Constants
        Materials = MaterialConstants(Tag=np.unique(CellTags.values),
                                      YoungsModulus=[1E4, 5E3, 2E4],
                                      PoissonRatio=[0.3, 0.3, 0.3])


        # Define lamé parameters as function of cell tags
        Lambda, Mu = LameParameters(Mesh, CellTags, Materials)

        # Compute surface
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
        S = np.zeros((3,3)) # Stiffness matrix
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

        
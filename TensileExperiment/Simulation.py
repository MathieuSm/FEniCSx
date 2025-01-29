#%% #!/usr/bin/env python3

Description = """
Script used to perform tensile test simulation using FEniCS
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2025'
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

def LowerSide(x):

    """
    Identifies and returns a boolean array indicating the vertices on the lower side of the mesh.
    
    Parameters:
    x (numpy.ndarray): The coordinates of the vertices.
    
    Returns:
    numpy.ndarray: A boolean array where True indicates the vertex is on the lower side.
    """

    return np.isclose(x[2], Geometry[:,2].min())
    
def UpperSide(x):

    """
    Identifies and returns a boolean array indicating the vertices on the upper side of the mesh.
    
    Parameters:
    x (numpy.ndarray): The coordinates of the vertices.
    
    Returns:
    numpy.ndarray: A boolean array where True indicates the vertex is on the upper side.
    """

    return np.isclose(x[2], Geometry[:,2].max())


#%% Main

def Main():

    # Define paths
    MeshPath = Path(__file__).parent / 'Meshes'
    OutputPath = Path(__file__).parent / 'Simulations'

    # List meshes
    Meshes = sorted([F for F in MeshPath.iterdir() if F.name.endswith('.msh')])

    # Test definition
    IniS = 1                                           # Initial state (-)
    FinS = 1.5                                         # Final state/stretch (-)
    NumberSteps = 10                                   # Number of steps (-)
    DeltaStretch = round((FinS-IniS)/NumberSteps,3)    # Stretch step (-)

    # Scan resolution (mm)
    Resolution = 0.028

    # Iterate over each ROI
    for m, MeshFile in enumerate(Meshes):

        # Print time
        Time.Process(1,MeshFile.name[:-4])

        # Read Mesh and create model
        InitializeGMSH()
        Mesh, CellTags, Classes = ReadMesh(MeshFile)

        # Record time
        Time.Process(1, MeshFile.name[:-4])

        # Define Material Constants
        E      = PETSc.ScalarType(1e4)        # Young's modulus (Pa)
        Nu     = PETSc.ScalarType(0.3)        # Poisson's ratio (-)
        Mu     = fem.Constant(Mesh, E/(2*(1 + Nu)))               # Shear modulus (kPa)
        Lambda = fem.Constant(Mesh, E*Nu/((1 + Nu)*(1 - 2*Nu)))   # First Lamé parameter (kPa)

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
            
            return Lambda * ufl.nabla_div(u) * I + 2 * Mu * Epsilon(u)
        Psi = ufl.inner(Sigma(u), Epsilon(v)) * ufl.dx

        # Boundary conditions
        Geometry = Mesh.geometry.x
        u_0 = fem.Constant(Mesh, PETSc.ScalarType((0,0,0)))   # No displacement
        u_1 = fem.Constant(Mesh, PETSc.ScalarType((0,0,0)))   # Applied displacement
        Bottom_DOFs = fem.locate_dofs_geometrical(V, LowerSide)
        Upper_DOFs = fem.locate_dofs_geometrical(V, UpperSide)
        BCl = fem.dirichletbc(u_0, Bottom_DOFs, V)
        BCu = fem.dirichletbc(u_1, Upper_DOFs, V)
        BCs = [BCl, BCu]

        # External loads
        f = fem.Constant(Mesh,(0.0, 0.0, 0.0))
        Load = ufl.dot(f, u) * ufl.ds

        # Model problem
        Problem = LinearProblem(Psi, Load, BCs, petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})

        # Solve for all loadcases
        Data = pd.DataFrame(columns=['Displacement [mm]','Force [N]'])
        Area = 1
        for t in range(NumberSteps+1):

            # Update displacement
            Displacement = t * DeltaStretch * (Geometry[:,2].max() - Geometry[:,2].min())
            u_1.value[2] = Displacement

            # Solve problem
            uh = Problem.solve()
            uh.name = 'Deformation'

            # Compute stress
            Te = ufl.TensorElement(ElementType, Mesh.ufl_cell(), 1)
            T = fem.FunctionSpace(Mesh, Te)
            Expression = fem.Expression(Sigma(uh), T.element.interpolation_points())
            Stress = fem.Function(T)
            Stress.interpolate(Expression)
            Stress.name = 'Stress'

            # Evaluate stress at the mid-section
            Zmid = (Geometry[:,2].max() - Geometry[:,2].min()) / 2
            P33 = Stress.eval(PETSc.ScalarType((0,0,Zmid)),0)[8]
            
            # Compute force
            Force = P33 * Area

            # Store results
            Data.loc[t,'Displacement [mm]'] = Displacement * Resolution
            Data.loc[t,'Force [N]'] = Force

        Data.to_csv(OutputPath / 'Results.csv')

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

        
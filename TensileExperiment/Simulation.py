#%% #!/usr/bin/env python3

Description = """
Script used to perform tensile test simulation using FEniCS
"""

__author__ = ['Mathieu Simon']
__date_created__ = '01-03-2025'
__date__ = '12-04-2024'
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

#%% Define geometric spaces

def LowerSide(x):
    return np.isclose(x[2], Geometry[:,2].min())
    
def UpperSide(x):
    return np.isclose(x[2], Geometry[:,2].max())

def KUBCs(E_Hom, Faces, Geometry, Mesh, V):

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

    for i, Vertice in enumerate(V_Bottom):

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

    for i, Vertice in enumerate(V_East):

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

    for i, Vertice in enumerate(V_North):

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

    for i, Vertice in enumerate(V_Top):

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
        if gmsh.is_initialized():
            gmsh.clear()
        else:
            gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 1)
        gmsh.merge(str(MeshFile))
        Mesh, Tags, Classes = io.gmshio.model_to_mesh(gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3)

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
            return 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
        def Sigma(u):
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
        FileName = OutputPath / MeshFile.name[:-4]
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

            Zmid = (Geometry[:,2].max() - Geometry[:,2].min()) / 2
            P33 = Stress.eval(PETSc.ScalarType((0,0,Zmid)),0)[8]
            Force = P33 * Area

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

        
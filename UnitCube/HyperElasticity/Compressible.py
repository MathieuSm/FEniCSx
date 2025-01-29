#%% #!/usr/bin/env python3

Description = """
This script performs a tensile test simulation using the FEniCSx library. 
It sets up a finite element model of a hyperelastic material, applies boundary conditions, 
and solves the resulting nonlinear system of equations and compare with theorical solution.
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
import matplotlib.pyplot as plt
from dolfinx import io, fem, mesh, plot
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem.petsc import NonlinearProblem

sys.path.append(str(Path(__file__).parents[2]))
from Utils import Time

#%% Define functions

def NeoHookean(Nu, Mu, Lambda1):

    """
    Computes uniaxial stress using Neo-Hookean strain energy density function.
    
    Parameters:
    Nu (float): Poisson's ratio.
    Mu (float): Shear modulus.
    Lambda1 (float): The elongation ratio.
    
    Returns:
    float: The Neo-Hookean uniaxial stress.
    """

    return Mu*Lambda1**(-2*Nu)*(Lambda1**(4*Nu + 4) - 1)/Lambda1**2

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

def PlotResults(V,uh):

    pv.start_xvfb()

    # Create plotter and pyvista grid
    PL = pv.Plotter(off_screen=True)
    Topology, Cell_types, Geometry = plot.vtk_mesh(V)
    Grid = pv.UnstructuredGrid(Topology, Cell_types, Geometry)

    sargs = dict(font_family='times', 
                width=0.05,
                height=0.75,
                vertical=True,
                position_x=0.85,
                position_y=0.125,
                title_font_size=30,
                label_font_size=20
                )

    # Attach vector values to grid and warp grid by vector
    Grid['Displacement'] = uh.x.array.reshape((Geometry.shape[0], 3))
    Actor = PL.add_mesh(Grid, style='wireframe', color=(0,0,0))
    Warped = Grid.warp_by_vector('Displacement', factor=1.0)
    Actor = PL.add_mesh(Warped, cmap='jet', show_edges=True, scalar_bar_args=sargs)
    
    PL.camera_position = 'xz'
    PL.camera.roll = 0
    PL.camera.elevation = 30
    PL.camera.azimuth = 30
    PL.camera.zoom(1.0)
    PL.show_axes()
    Array = PL.screenshot(Path(__file__).parents[1]/'Results.png', scale=2)

    return

#%% Main

def Main():

    # Test definition
    IniS = 0.1                                           # Initial state (-)
    FinS = 5.0                                         # Final state/stretch (-)
    NumberSteps = 50                                   # Number of steps (-)
    DeltaStretch = round((FinS-IniS)/NumberSteps,3)    # Stretch step (-)
    Streches = np.round(np.arange(IniS,FinS+DeltaStretch,DeltaStretch),1)

    # Generate mesh
    Mesh, CellTags, Classes = io.gmshio.read_from_msh('../Cube.msh', comm=MPI.COMM_WORLD, rank=0, gdim=3)
    Height = Mesh.geometry.x[:,2].max() - Mesh.geometry.x[:,2].min()

    # Define Material Constants
    Lambda = 1e6                        # First Lamé parameter (Pa)
    Mu = 660                            # Shear modulus (Pa)
    E = Mu*(3*Lambda+2*Mu)/(Lambda+Mu)  # Young's modulus (Pa)
    Nu = Lambda / (2*(Lambda+Mu))       # Poisson's ratio (-)
    # Note that Poisson's ratio nu is very close to 0.5

    # Functions space over the mesh domain
    ElementType = 'Lagrange'
    PolDegree = 1
    Ve = ufl.VectorElement(ElementType, Mesh.ufl_cell(), PolDegree)
    V = fem.FunctionSpace(Mesh, Ve)
    u = fem.Function(V)     # Incremental displacement
    v = ufl.TestFunction(V)      # Test function

    # Kinematics
    d = len(u)                         # Spatial dimension
    I = ufl.variable(ufl.Identity(d))  # Identity tensor

    # Deformation gradient
    F = ufl.variable(I + ufl.grad(u))

    # Right Cauchy-Green tensor
    C = ufl.variable(F.T * F)

    # Invariants of deformation tensors
    Ic = ufl.variable(ufl.tr(C))
    J = ufl.variable(ufl.det(F))

    # Stored strain energy density (compressible neo-Hookean model)
    Psi = (Mu / 2) * (Ic - 3) - Mu * ufl.ln(J) + (Lambda / 2) * (J-1)**2

    # Hyper-elasticity
    P = ufl.diff(Psi, F) # 1st Piola–Kirchhoff stress tensor (nominal stress)
    S = ufl.inv(F) * ufl.diff(Psi, F) # 2nd Piola–Kirchhoff stress tensor
    Sigma = 1/ufl.det(F) * ufl.diff(Psi, F) * F.T # Cauchy stress (True of material stress)

    # External forces
    T = fem.Constant(Mesh, PETSc.ScalarType((0, 0, 0)))

    # Balance
    Fpi = ufl.inner(ufl.grad(v), P) * ufl.dx - ufl.inner(v, T) * ufl.ds
    
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

    # Normal to top surface
    n = fem.Constant(Mesh, PETSc.ScalarType((0,0,1)))

    # Deformation direction
    Dir = fem.Constant(Mesh, PETSc.ScalarType((0,0,1)))

    # Model problem
    Problem = NonlinearProblem(Fpi, u, BCs)
    Solver = NewtonSolver(Mesh.comm, Problem)

    # Solve for all loadcases
    Data = pd.DataFrame()
    Js = np.zeros(NumberSteps+1)
    for t, Strain in enumerate(Streches):

        # Update displacement
        Displacement = (Strain-1) * Height
        u1.value = Displacement

        # Solve problem
        Solver.solve(u)

        # Compute stress
        Te = ufl.TensorElement(ElementType, Mesh.ufl_cell(), 1)
        T = fem.FunctionSpace(Mesh, Te)
        Expression = fem.Expression(Sigma, T.element.interpolation_points())
        Stress = fem.Function(T)
        Stress.interpolate(Expression)
        Stress.name = 'Stress'

        # Evaluate stress at the center of the cube
        P33 = Stress.eval(PETSc.ScalarType((0.5,0.5,0.5)),0)[8]

        Data.loc[t,'Strain'] = Strain
        Data.loc[t,'Stress'] = P33

    ModelStress = NeoHookean(Nu, Lambda, Streches)
    Theory = ModelStress

    # Plot results
    Figure, Axis = plt.subplots(1,1)
    Axis.plot(Data['Strain'], Theory, color=(1,0,0), linestyle='none', marker='o', fillstyle='none', label='Theory')
    Axis.plot(Data['Strain'], Data['Stress'], color=(0,0,1), label='Simulation')
    Axis.set_xlabel('Strain (-)')
    Axis.set_ylabel('Stress (Pa)')
    plt.legend()
    plt.show(Figure)

    PlotResults(V,u)

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

        
#%% #!/usr/bin/env python3

Description = """
This script sets up and solves a linear elasticity problem on a unit cube mesh using the FEniCSx library.
It defines the material properties, boundary conditions, and solves for the displacement field under tensile loading.
The results are then compared to theory and visualized using pyvista.
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
from dolfinx.fem.petsc import LinearProblem

sys.path.append(str(Path(__file__).parents[1]))
from Utils import Time

#%% Define geometric spaces

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
    IniS = 1                                           # Initial state (-)
    FinS = 1.05                                        # Final state/stretch (-)
    NumberSteps = 10                                   # Number of steps (-)
    DeltaStretch = round((FinS-IniS)/NumberSteps,3)    # Stretch step (-)

    # Generate mesh
    Mesh, CellTags, Classes = io.gmshio.read_from_msh('Cube.msh', comm=MPI.COMM_WORLD, rank=0, gdim=3)
    Area = 1 * 1
    Height = 1

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

    # External loads
    f = fem.Constant(Mesh,(0.0, 0.0, 0.0))
    Load = ufl.dot(f, u) * ufl.ds

    # Model problem
    Problem = LinearProblem(Psi, Load, BCs, petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})

    # Solve for all loadcases
    Data = pd.DataFrame()
    for t in range(NumberSteps+1):

        # Update displacement
        Displacement = t * DeltaStretch * Height
        u1.value = Displacement

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

        # Evaluate stress at the center of the cube
        P33 = Stress.eval(PETSc.ScalarType((0.5,0.5,0.5)),0)[8]
        Force = P33 * Area

        Data.loc[t,'Displacement'] = Displacement
        Data.loc[t,'Simulation'] = Force
        Data.loc[t,'Theory'] = E * Displacement


    # Plot results
    Figure, Axis = plt.subplots(1,1)
    Axis.plot(Data['Displacement'], Data['Theory'], color=(1,0,0), label='Theory')
    Axis.plot(Data['Displacement'], Data['Simulation'], color=(0,0,1), label='Simulation')
    Axis.set_xlabel('Displacement')
    Axis.set_ylabel('Force')
    plt.legend()
    plt.show(Figure)

    PlotResults(V,uh)

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

        
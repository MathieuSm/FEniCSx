#%% #!/usr/bin/env python3

Description = """
Script used to perform cube homogenization using FEniCSx
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
from dolfinx import io, fem, mesh, plot
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

#%% Periodicity-Mixed Uniform Boundary Conditions

def Tensile1BC(Value, L1, Faces, Mesh, V):

    """
    Applies tensile boundary conditions in the first direction.
    
    Parameters:
    Value (float): The value of the tensile strain.
    L1 (float): The length in the first direction.
    Faces (list): List of faces at the boundaries.
    Mesh (dolfinx.mesh.Mesh): The mesh object.
    V (dolfinx.fem.FunctionSpace): The function space.
    
    Returns:
    list: A list of Dirichlet boundary conditions.
    """

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u1 = fem.Constant(Mesh,(Value*L1/2))
    u2 = fem.Constant(Mesh,(-Value*L1/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(2), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(2), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(1), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(1), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(0), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(0), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u0, D_Top, V.sub(2))
    BC_Bottom = fem.dirichletbc(u0, D_Bottom, V.sub(2))
    BC_North = fem.dirichletbc(u0, D_North, V.sub(1))
    BC_South = fem.dirichletbc(u0, D_South, V.sub(1))
    BC_East = fem.dirichletbc(u1, D_East, V.sub(0))
    BC_West = fem.dirichletbc(u2, D_West, V.sub(0))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Tensile2BC(Value, L2, Faces, Mesh, V):

    """
    Applies tensile boundary conditions in the second direction.
    
    Parameters:
    Value (float): The value of the tensile strain.
    L2 (float): The length in the second direction.
    Faces (list): List of faces at the boundaries.
    Mesh (dolfinx.mesh.Mesh): The mesh object.
    V (dolfinx.fem.FunctionSpace): The function space.
    
    Returns:
    list: A list of Dirichlet boundary conditions.
    """

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u1 = fem.Constant(Mesh,(Value*L2/2))
    u2 = fem.Constant(Mesh,(-Value*L2/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(2), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(2), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(1), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(1), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(0), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(0), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u0, D_Top, V.sub(2))
    BC_Bottom = fem.dirichletbc(u0, D_Bottom, V.sub(2))
    BC_North = fem.dirichletbc(u1, D_North, V.sub(1))
    BC_South = fem.dirichletbc(u2, D_South, V.sub(1))
    BC_West = fem.dirichletbc(u0, D_West, V.sub(0))
    BC_East = fem.dirichletbc(u0, D_East, V.sub(0))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Tensile3BC(Value, L3, Faces, Mesh, V):

    """
    Applies tensile boundary conditions in the third direction.
    
    Parameters:
    Value (float): The value of the tensile strain.
    L3 (float): The length in the third direction.
    Faces (list): List of faces at the boundaries.
    Mesh (dolfinx.mesh.Mesh): The mesh object.
    V (dolfinx.fem.FunctionSpace): The function space.
    
    Returns:
    list: A list of Dirichlet boundary conditions.
    """

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u1 = fem.Constant(Mesh,(Value*L3/2))
    u2 = fem.Constant(Mesh,(-Value*L3/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(2), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(2), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(1), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(1), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(0), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(0), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u1, D_Top, V.sub(2))
    BC_Bottom = fem.dirichletbc(u2, D_Bottom, V.sub(2))
    BC_North = fem.dirichletbc(u0, D_North, V.sub(1))
    BC_South = fem.dirichletbc(u0, D_South, V.sub(1))
    BC_West = fem.dirichletbc(u0, D_West, V.sub(0))
    BC_East = fem.dirichletbc(u0, D_East, V.sub(0))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Shear12BC(Value, L1, L2, Faces, Mesh, V):

    """
    Applies shear boundary conditions in the 1-2 plane.
    
    Parameters:
    Value (float): The value of the shear strain.
    L1 (float): The length in the first direction.
    L2 (float): The length in the second direction.
    Faces (list): List of faces at the boundaries.
    Mesh (dolfinx.mesh.Mesh): The mesh object.
    V (dolfinx.fem.FunctionSpace): The function space.
    
    Returns:
    list: A list of Dirichlet boundary conditions.
    """

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u23 = fem.Constant(Mesh,(Value*L1/2))
    u32 = fem.Constant(Mesh,(-Value*L1/2))
    u13 = fem.Constant(Mesh,(Value*L2/2))
    u31 = fem.Constant(Mesh,(-Value*L2/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(2), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(2), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(0), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(0), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(1), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(1), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u0, D_Top, V.sub(2))
    BC_Bottom = fem.dirichletbc(u0, D_Bottom, V.sub(2))
    BC_North = fem.dirichletbc(u13, D_North, V.sub(0))
    BC_South = fem.dirichletbc(u31, D_South, V.sub(0))
    BC_East = fem.dirichletbc(u23, D_East, V.sub(1))
    BC_West = fem.dirichletbc(u32, D_West, V.sub(1))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Shear13BC(Value, L1, L3, Faces, Mesh, V):

    """
    Applies shear boundary conditions in the 1-3 plane.
    
    Parameters:
    Value (float): The value of the shear strain.
    L1 (float): The length in the first direction.
    L3 (float): The length in the third direction.
    Faces (list): List of faces at the boundaries.
    Mesh (dolfinx.mesh.Mesh): The mesh object.
    V (dolfinx.fem.FunctionSpace): The function space.
    
    Returns:
    list: A list of Dirichlet boundary conditions.
    """

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u23 = fem.Constant(Mesh,(Value*L1/2))
    u32 = fem.Constant(Mesh,(-Value*L1/2))
    u12 = fem.Constant(Mesh,(Value*L3/2))
    u21 = fem.Constant(Mesh,(-Value*L3/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(0), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(0), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(1), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(1), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(2), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(2), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u12, D_Top, V.sub(0))
    BC_Bottom = fem.dirichletbc(u21, D_Bottom, V.sub(0))
    BC_North = fem.dirichletbc(u0, D_North, V.sub(1))
    BC_South = fem.dirichletbc(u0, D_South, V.sub(1))
    BC_East = fem.dirichletbc(u23, D_East, V.sub(2))
    BC_West = fem.dirichletbc(u32, D_West, V.sub(2))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def Shear23BC(Value, L2, L3, Faces, Mesh, V):

    """
    Applies shear boundary conditions in the 2-3 plane.
    
    Parameters:
    Value (float): The value of the shear strain.
    L2 (float): The length in the second direction.
    L3 (float): The length in the third direction.
    Faces (list): List of faces at the boundaries.
    Mesh (dolfinx.mesh.Mesh): The mesh object.
    V (dolfinx.fem.FunctionSpace): The function space.
    
    Returns:
    list: A list of Dirichlet boundary conditions.
    """

    # Displacements
    u0 = fem.Constant(Mesh,(0.0))
    u12 = fem.Constant(Mesh,(Value*L3/2))
    u21 = fem.Constant(Mesh,(-Value*L3/2))
    u13 = fem.Constant(Mesh,(Value*L2/2))
    u31 = fem.Constant(Mesh,(-Value*L2/2))

    # Detect DOFs
    F_Bottom, F_Top, F_North, F_South, F_East, F_West = Faces
    D_Top = fem.locate_dofs_topological(V.sub(1), 0, F_Top)
    D_Bottom = fem.locate_dofs_topological(V.sub(1), 0, F_Bottom)
    D_North = fem.locate_dofs_topological(V.sub(2), 0, F_North)
    D_South = fem.locate_dofs_topological(V.sub(2), 0, F_South)
    D_East = fem.locate_dofs_topological(V.sub(0), 0, F_East)
    D_West = fem.locate_dofs_topological(V.sub(0), 0, F_West)

    # Apply boundary conditions
    BC_Top = fem.dirichletbc(u12, D_Top, V.sub(1))
    BC_Bottom = fem.dirichletbc(u21, D_Bottom, V.sub(1))
    BC_North = fem.dirichletbc(u13, D_North, V.sub(2))
    BC_South = fem.dirichletbc(u31, D_South, V.sub(2))
    BC_East = fem.dirichletbc(u0, D_East, V.sub(0))
    BC_West = fem.dirichletbc(u0, D_West, V.sub(0))

    return BC_Top, BC_Bottom, BC_North, BC_South, BC_East, BC_West

def PMUBC(E_Hom, Faces, L1, L2, L3, Mesh, V):

    """
    Applies periodicity-mixed uniform boundary conditions (PMUBC).
    
    Parameters:
    E_Hom (numpy.ndarray): The homogenized strain tensor.
    Faces (list): List of faces at the boundaries.
    L1 (float): The length in the first direction.
    L2 (float): The length in the second direction.
    L3 (float): The length in the third direction.
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

    if E_Hom[0,0] != 0:
        BCs = Tensile1BC(E_Hom[0,0], L1, Faces, Mesh, V)
    elif E_Hom[1,1] != 0:
        BCs = Tensile2BC(E_Hom[1,1], L2, Faces, Mesh, V)
    elif E_Hom[2,2] != 0:
        BCs = Tensile3BC(E_Hom[2,2], L3, Faces, Mesh, V)
    elif E_Hom[0,1] != 0:
        BCs = Shear12BC(E_Hom[0,1], L1, L2, Faces, Mesh, V)
    elif E_Hom[0,2] != 0:
        BCs = Shear13BC(E_Hom[0,2], L1, L3, Faces, Mesh, V)
    elif E_Hom[1,2] != 0:
        BCs = Shear23BC(E_Hom[1,2], L2, L3, Faces, Mesh, V)

    return BCs

#%%
# #%% Define pyvista plot

# def PyVistaPlot(Displacement, Value, FileName):

#     # Build mesh
#     Topology, Cells, Geometry = plot.vtk_mesh(Displacement.function_space)
#     FunctionGrid = pv.UnstructuredGrid(Topology, Cells, Geometry)

#     # Get deformation values
#     N = len(Displacement)
#     Deformation = np.zeros((Geometry.shape[0], 3))
#     Deformation[:, :N] = Displacement.x.array.reshape(Geometry.shape[0], N)

#     # Set deformation on mesh
#     FunctionGrid['Deformation'] = Deformation
#     FunctionGrid.set_active_vectors('Deformation')

#     # Warp mesh by deformation
#     Warped = FunctionGrid.warp_by_vector('Deformation', factor=1/Value)
#     Warped.set_active_vectors('Deformation')

#     # Arguments for the colorbar
#     Args = dict(font_family='times', 
#                 width=0.05,
#                 height=0.75,
#                 vertical=True,
#                 position_x=0.9,
#                 position_y=0.125,
#                 title_font_size=30,
#                 label_font_size=20
#                 )

#     # Plot with pyvista
#     pl = pv.Plotter(off_screen=True)
#     pl.add_mesh(Warped, cmap='jet')#, scalar_bar_args=Args)
#     pl.camera_position = 'xz'
#     pl.camera.roll = 0
#     pl.camera.elevation = 30
#     pl.camera.azimuth = 30
#     pl.camera.zoom(1.0)
#     pl.add_axes(viewport=(0,0,0.25,0.25))
#     pl.screenshot(FileName, return_img=False)
#     # pl.show()

#     return

#%% Main

def Main(BCsType='KUBC'):

    # Define paths
    MeshPath = Path(__file__).parent / 'Meshes'
    OutputPath = Path(__file__).parent / 'FEniCS'

    # List meshes
    Meshes = sorted([F for F in MeshPath.iterdir() if F.name.endswith('.msh')])

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
            
            return Lambda * ufl.nabla_div(u) * I + 2 * Mu * Epsilon(u)
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
        FileName = OutputPath / MeshFile.name[:-4]
        S = np.zeros((6,6)) # Stiffness matrix
        for LoadCase in range(6):

            Time.Update(LoadCase/6, LCs[LoadCase])

            E_Hom = E_Homs[LoadCase]
            if BCsType == 'KUBC':
                BCs = KUBCs(E_Hom, Vertices, Geometry, Mesh, V)
            elif BCsType == 'PMUBC':
                BCs = PMUBC(E_Hom, Faces, L1, L2, L3, Mesh, V)

            # Solve problem
            Problem = LinearProblem(Psi, Load, BCs, petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg'})
            uh = Problem.solve()
            uh.name = 'Deformation'

            # Plot results
            # PyVistaPlot(uh, Value, str(FileName) + '.png')

            # # Compute stress
            # Te = ufl.TensorElement(ElementType, Mesh.ufl_cell(), 1)
            # T = fem.FunctionSpace(Mesh, Te)
            # Expression = fem.Expression(Sigma(uh), T.element.interpolation_points())
            # Stress = fem.Function(T)
            # Stress.interpolate(Expression)
            # Stress.name = 'Stress'

            # # Store results in vtk file
            # with io.VTXWriter(Mesh.comm, str(FileName) + '_' + LCs[LoadCase] + '.bp', [uh, Stress]) as vf:
            #     vf.write(0)

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

        
#%% !/usr/bin/env python3

"""
This module contains classes and functions for generating 2D and 3D meshes from numpy arrays using gmsh.
It includes mapping functions for 2D and 3D arrays, a Time class for tracking processing time, and a Mesh class for mesh generation.
"""

__author__ = ['Mathieu Simon']
__date_created__ = '26-01-2024'
__license__ = 'MIT'
__version__ = '1.0'


import sys
import gmsh
import time
import argparse
import numpy as np
from numba import njit
from pathlib import Path
import SimpleITK as sitk

#%% Mapping functions
@njit
def Mapping2D(Array: np.array):
    
    """
    Maps a 2D numpy array to nodes and elements.

    Parameters:
    Array (np.array): 2D numpy array to be mapped.

    Returns:
    tuple: Nodes, Coords, Elements, ElementsNodes.
    """
    
    X, Y = Array.T.shape

    # Generate nodes map
    Index = 0
    Nodes = np.zeros((Y+1,X+1),'int')
    Coords = np.zeros((Y+1,X+1,2),'int')
    for Yn in range(Y + 1):
        for Xn in range(X + 1):
            Index += 1
            Nodes[Yn,Xn] = Index
            Coords[Yn,Xn] = [Yn, Xn]

    # Generate elements map
    Index = 0
    Elements = np.zeros((Y, X),'int')
    ElementsNodes = np.zeros((Y, X, 4), 'int')
    for Xn in range(X):
            for Yn in range(Y):
                Index += 1
                Elements[Yn, Xn] = Index
                ElementsNodes[Yn, Xn, 0] = Nodes[Yn, Xn]
                ElementsNodes[Yn, Xn, 1] = Nodes[Yn, Xn+1]
                ElementsNodes[Yn, Xn, 2] = Nodes[Yn+1, Xn+1]
                ElementsNodes[Yn, Xn, 3] = Nodes[Yn+1, Xn]

    return Nodes, Coords, Elements, ElementsNodes

@njit
def Mapping3D(Array: np.array):
    
    """
    Maps a 3D numpy array to nodes and elements.

    Parameters:
    Array (np.array): 3D numpy array to be mapped.

    Returns:
    tuple: Nodes, Coords, Elements, ElementsNodes.
    """
    
    X, Y, Z = Array.T.shape

    # Generate nodes map
    Index = 0
    Nodes = np.zeros((Z+1,Y+1,X+1),'int')
    Coords = np.zeros((Z+1,Y+1,X+1,3),'int')
    for Zn in range(Z + 1):
        for Yn in range(Y + 1):
            for Xn in range(X + 1):
                Index += 1
                Nodes[Zn,Yn,Xn] = Index
                Coords[Zn,Yn,Xn] = [Zn, Yn, Xn]

    # Generate elements map
    Index = 0
    Elements = np.zeros((Z, Y, X),'int')
    ElementsNodes = np.zeros((Z, Y, X, 8), 'int')
    for Xn in range(X):
            for Yn in range(Y):
                for Zn in range(Z):
                    Index += 1
                    Elements[Zn, Yn, Xn] = Index
                    ElementsNodes[Zn, Yn, Xn, 0] = Nodes[Zn, Yn, Xn]
                    ElementsNodes[Zn, Yn, Xn, 1] = Nodes[Zn, Yn, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 2] = Nodes[Zn, Yn+1, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 3] = Nodes[Zn, Yn+1, Xn]
                    ElementsNodes[Zn, Yn, Xn, 4] = Nodes[Zn+1, Yn, Xn]
                    ElementsNodes[Zn, Yn, Xn, 5] = Nodes[Zn+1, Yn, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 6] = Nodes[Zn+1, Yn+1, Xn+1]
                    ElementsNodes[Zn, Yn, Xn, 7] = Nodes[Zn+1, Yn+1, Xn]

    return Nodes, Coords, Elements, ElementsNodes

#%% Mesh class
class Mesh():
    
    """
    Class to generate and clean 2D or 3D meshes from numpy arrays.
    """

    def __init__(self):
        pass

    def CleanAndSort(self, Array, Nodes, Coords, Elements, ElementsNodes):

        """
        Clean and sort the mesh by removing unnecessary elements.

        Parameters:
        Array (np.array): Input array.
        Nodes (np.array): Nodes array.
        Coords (np.array): Coordinates array.
        Elements (np.array): Elements array.
        ElementsNodes (np.array): Elements nodes array.

        Returns:
        tuple: Cleaned ElementsNodes, Elements, Coords, Nodes.
        """
        
        NodesNeeded = np.unique(ElementsNodes[Array.astype(bool)])

        ElementsNodes = ElementsNodes[Array.astype(bool)]
        Elements = Elements[Array.astype(bool)]
        Coords = Coords[np.isin(Nodes,NodesNeeded)]
        Nodes = Nodes[np.isin(Nodes,NodesNeeded)]

        return ElementsNodes, Elements, Coords, Nodes

    def Generate(self, Array:np.array, FName:'Mesh.msh'):

        """
        Generate a mesh from the given input array.

        Parameters:
        Array (np.array): Input array.
        FName (str): Filename for the generated mesh.
        """
        
        Dim = len(Array.shape)
        if Dim == 2:

            # Perform mapping
            Nodes, Coords, Elements, ElementsNodes = Mapping2D(Array)

        elif Dim == 3:

            # Perform mapping
            Nodes, Coords, Elements, ElementsNodes = Mapping3D(Array)

        else:
            print('Dimension more implemented (only 2D or 3D array)')
            return

        # Identify the different phases
        Phases = np.unique(Array)[1:]
        Phase_Elements = []
        for Phase in Phases:
            Phase_Elements.append(Elements[Array == Phase])

        # Remove 0 elements
        ElementsNodes, Elements, Coords, Nodes = self.CleanAndSort(Array, Nodes, Coords, Elements, ElementsNodes)

        # Generate tags
        NodeTags = np.arange(len(Nodes)) + 1
        ElementsTags = np.arange(len(Elements)) + 1
        NodesArgSorted = np.argsort(Nodes)
        ElementsNodes = np.searchsorted(Nodes[NodesArgSorted], ElementsNodes)
        PhysicalTags = np.ones(len(Elements),int)
        for P, Phase in enumerate(Phases[1:]):
            PhysicalTags[[E in Phase_Elements[P+1] for E in Elements]] = Phase

        # Generate mesh
        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 1)

        # Phase entity
        for Phase in Phases:
            gmsh.model.addDiscreteEntity(Dim, Phase)

        # Nodes
        if Dim == 2:
            ZCoords = np.hstack([Coords, np.zeros((len(Coords),1),int)])
            NodesCoords = [C for C in ZCoords.ravel()]
        else:
            NodesCoords = [C for C in Coords.ravel()]
        gmsh.model.mesh.addNodes(Dim, 1, list(NodeTags), NodesCoords)

        # Element
        for Phase in Phases:
            if Dim == 2:
                ElementType = list(np.zeros(sum(PhysicalTags==Phase),int)+3)
            else:
                ElementType = list(np.zeros(sum(PhysicalTags==Phase),int)+5)
            TagList = [[E] for E in ElementsTags[PhysicalTags==Phase]]
            NodeList = [list(N+1) for N in ElementsNodes[PhysicalTags==Phase]]
            gmsh.model.mesh.addElements(Dim, Phase, ElementType, TagList, NodeList)

        # Physical group
        for Phase in Phases:
            gmsh.model.addPhysicalGroup(Dim, [Phase], Phase)

        # Write mesh
        gmsh.write(FName)
        gmsh.finalize()

        return

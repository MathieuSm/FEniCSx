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

#%% Time class
class Time():
    
    """
    Class to measure and display the processing time.
    """
    
    def __init__(self):
        self.Width = 15
        self.Length = 16
        self.Text = 'Process'
        self.Tic = time.time()
    
    def Set(self, Tic=None):
        
        """
        Set the start time.

        Parameters:
        Tic (float): Start time. Defaults to current time.
        """
        
        if Tic == None:
            self.Tic = time.time()
        else:
            self.Tic = Tic

    def Print(self, Tic=None,  Toc=None):
        
        """
        Print elapsed time in HH:MM:SS format.

        Parameters:
        Tic (float): Start time. Defaults to self.Tic.
        Toc (float): End time. Defaults to current time.
        """
        
        if Tic == None:
            Tic = self.Tic
            
        if Toc == None:
            Toc = time.time()


        Delta = Toc - Tic

        Hours = np.floor(Delta / 60 / 60)
        Minutes = np.floor(Delta / 60) - 60 * Hours
        Seconds = Delta - 60 * Minutes - 60 * 60 * Hours

        print('\nProcess executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))

        return

    def Update(self, Progress, Text=''):

        """
        Update the progress bar.

        Parameters:
        Progress (float): Progress fraction (0 to 1).
        Text (str): Text to display. Defaults to self.Text.
        """

        Percent = int(round(Progress * 100))
        Np = self.Width * Percent // 100
        Nb = self.Width - Np

        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        Ns = self.Length - len(Text)
        if Ns >= 0:
            Text += Ns*' '
        else:
            Text = Text[:self.Length]
        
        Line = '\r' + Text + ' [' + Np*'=' + Nb*' ' + ']' + f' {Percent:.0f}%'
        print(Line, sep='', end='', flush=True)

    def Process(self, StartStop:bool, Text=''):

        """
        Start or stop the process timer and print progress.

        Parameters:
        StartStop (bool): True to start, False to stop.
        Text (str): Text to display. Defaults to self.Text.
        """

        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        if StartStop*1 == 1:
            self.Tic = time.time()
            self.Update(0, Text)

        elif StartStop*1 == 0:
            self.Update(1, Text)
            self.Print()

Time = Time()

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
            Time.Process(1,'Generate 2D Mesh')

            # Perform mapping
            Nodes, Coords, Elements, ElementsNodes = Mapping2D(Array)
            Time.Update(1/5,'Nodes Map Done')

        elif Dim == 3:
            Time.Process(1,'Generate 3D Mesh')

            # Perform mapping
            Nodes, Coords, Elements, ElementsNodes = Mapping3D(Array)
            Time.Update(1/5,'Nodes Map Done')

        else:
            print('Dimension more implemented (only 2D or 3D array)')
            return

        # Identify the different phases
        Phases = np.unique(Array)[1:]
        Phase_Elements = []
        for Phase in Phases:
            Phase_Elements.append(Elements[Array == Phase])
        Time.Update(2/6,'Phases attributed')

        # Remove 0 elements
        ElementsNodes, Elements, Coords, Nodes = self.CleanAndSort(Array, Nodes, Coords, Elements, ElementsNodes)
        Time.Update(3/6,'0 elements removed')

        # Generate tags
        NodeTags = np.arange(len(Nodes)) + 1
        ElementsTags = np.arange(len(Elements)) + 1
        NodesArgSorted = np.argsort(Nodes)
        ElementsNodes = np.searchsorted(Nodes[NodesArgSorted], ElementsNodes)
        PhysicalTags = np.ones(len(Elements),int)
        for P, Phase in enumerate(Phases[1:]):
            PhysicalTags[[E in Phase_Elements[P+1] for E in Elements]] = Phase
        Time.Update(4/6,'Tags attributed')

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
        Time.Update(5/6,'Nodes added')

        # Element
        for Phase in Phases:
            if Dim == 2:
                ElementType = list(np.zeros(sum(PhysicalTags==Phase),int)+3)
            else:
                ElementType = list(np.zeros(sum(PhysicalTags==Phase),int)+5)
            TagList = [[E] for E in ElementsTags[PhysicalTags==Phase]]
            NodeList = [list(N+1) for N in ElementsNodes[PhysicalTags==Phase]]
            gmsh.model.mesh.addElements(Dim, Phase, ElementType, TagList, NodeList)
        Time.Update(6/6,'Elements added')

        # Physical group
        for Phase in Phases:
            gmsh.model.addPhysicalGroup(Dim, [Phase], Phase)

        # Write mesh
        gmsh.write(FName)
        gmsh.finalize()

        # Print time
        Time.Process(0,'Mesh written')

        return

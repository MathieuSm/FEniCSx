#%% !/usr/bin/env python3

Description = """
Script used to write 2D square elements using gmsh
"""
__author__ = ['Mathieu Simon']
__date_created__ = '26-01-2025'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import sys
import argparse
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from skimage import io, morphology

sys.path.append(str(Path(__file__).parents[2]))
from Utils import Mesh


#%% Main
def Main():

    # Define paths
    ROIsPath = Path(__file__).parent / 'ROIs'
    OutputPath = Path(__file__).parent / 'Meshes'

    # List ROIs
    ROIs = sorted([F for F in ROIsPath.iterdir() if F.name.endswith('.png')])

    # Iterate over each ROI
    for i, ROI in enumerate(ROIs):

        # Read scan
        Array = io.imread(ROIsPath / ROI)

        # Reattribute phases tags
        Phases = np.unique(Array)
        for P, Phase in enumerate(Phases):
            Array[Array == Phase] = P

        # Generate mesh
        FileName = str(OutputPath / (ROI.name[:-4] + '.msh'))
        Mesh.Generate(Array, FileName)

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
#%%
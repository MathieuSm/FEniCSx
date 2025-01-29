#%% !/usr/bin/env python3

Description = """
This script reads ISQ files, processes the scans by cropping, resampling, filtering noise, and segmenting the images.
The processed data is then used to generate hexahedral mesh elements.
"""
__author__ = ['Mathieu Simon']
__date_created__ = '25-01-2025'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import sys
import argparse
import numpy as np
from pathlib import Path
from skimage import filters, morphology

sys.path.append(str(Path(__file__).parents[1]))
from Utils import Read, Mesh

#%% Main
def Main():

    # Define paths
    ScansPath = Path(__file__).parent / 'Scans'
    OutputPath = Path(__file__).parent / 'Meshes'

    # List ROIs
    Scans = sorted([F for F in ScansPath.iterdir() if F.name.endswith('.ISQ')])

    # Iterate over each ROI
    for i, Scan in enumerate(Scans):

        # Read scan
        Array, AddData = Read.ISQ(str(Scan))

        # Crop scan
        Cropped = Array[:,900:1210,855:1155]

        # Resample scan
        Resampled = Cropped[::4,::4,::4]

        # Filter noise
        Filtered = filters.gaussian(Resampled, sigma=1)

        # Segment image
        Otsu = filters.threshold_otsu(Filtered)
        Segmented = Filtered > Otsu

        # Clean segmentation
        Labels = morphology.label(Segmented)
        Max = 0
        for L in range(np.max(Labels)+1):
            Sum = np.sum(Labels == L)
            if Sum > Max:
                Max = Sum
                Label = L
        Cleaned = (Labels == Label) * 1

        # Generate mesh
        FileName = str(OutputPath / (Scan.name[:-4] + '.msh'))
        Mesh.Generate(Cleaned, FileName)

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
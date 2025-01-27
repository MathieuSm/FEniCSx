#%% !/usr/bin/env python3

Description = """
Init file to initialize the different utility script
"""

# File info
__author__ = ['Mathieu Simon']
__date_created__ = '05-12-2024'
__date__ = '26-01-2025'
__license__ = 'MIT'
__version__ = '1.0'

# Import general modules
import sys
import numpy as np
from pathlib import Path

# Import custom modules
sys.path.append(str(Path(__file__).parent))
from Time import Time
from Plot import Plot
from Read import Read
from Mesh import Mesh
from Image import Image
from Tensor import Tensor

# Initialize classes
Time = Time()
Plot = Plot()
Read = Read()
Mesh = Mesh()
Image = Image()
Tensor = Tensor()

# Different print options
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
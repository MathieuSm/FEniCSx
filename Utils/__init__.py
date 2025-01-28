#%% !/usr/bin/env python3

"""
This module initializes the different utility scripts for the application.
"""

# File info
__author__ = ['Mathieu Simon']
__date_created__ = '05-12-2024'
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

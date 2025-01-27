#%% !/usr/bin/env python3

Description = """
Class to perform image manipulations
"""

__author__ = ['Mathieu Simon']
__date_created__ = '07-12-2024'
__date__ = '07-12-2024'
__license__ = 'MIT'
__version__ = '1.0'


import numpy as np
import SimpleITK as sitk

class Image():

    def __init__(self):
        pass

    def Resample(self, Image, Factor=None, Size=[None], Spacing=[None], Order=1):

        """
        Resample a SimpleITK image by either a given factor, a new size, or
        a new voxel spacing. Order stands for interpolation order e.g.
        Order = 1: Linear interpolation 
        """

        Dimension = Image.GetDimension()
        OriginalSpacing = np.array(Image.GetSpacing())
        OriginalSize = np.array(Image.GetSize())
        PhysicalSize = OriginalSize * OriginalSpacing

        Origin = Image.GetOrigin()
        Direction = Image.GetDirection()

        if Factor:
            NewSize = [round(Size/Factor) for Size in Image.GetSize()] 
            NewSpacing = [PSize/(Size-1) for Size,PSize in zip(NewSize, PhysicalSize)]
        
        elif Size[0]:
            NewSize = Size
            NewSpacing = [PSize/Size for Size, PSize in zip(NewSize, PhysicalSize)]
        
        elif Spacing[0]:
            NewSpacing = Spacing
            NewSize = [np.floor(Size/Spacing).astype('int') + 1 for Size,Spacing in zip(PhysicalSize, NewSpacing)]

        NewArray = np.zeros(NewSize[::-1],'int')
        NewImage = sitk.GetImageFromArray(NewArray)
        NewImage.SetOrigin(Origin - OriginalSpacing/2)
        NewImage.SetDirection(Direction)
        NewImage.SetSpacing(NewSpacing)
    
        Transform = sitk.TranslationTransform(Dimension)
        Resampled = sitk.Resample(Image, NewImage, Transform, Order)
        
        return Resampled


#%% !/usr/bin/env python3

Description = """
This module contains a class for performing image manipulations using SimpleITK.
"""

__author__ = ['Mathieu Simon']
__date_created__ = '07-12-2024'
__license__ = 'MIT'
__version__ = '1.0'


import numpy as np
import SimpleITK as sitk

class Image():

    """
    Class for performing various image manipulations using SimpleITK.
    """

    def __init__(self):
        """
        Initialize the Image class.
        """
        pass

    def Resample(self, Image, Factor=None, Size=[None], Spacing=[None], Order=1):

        """
        Resample a SimpleITK image by either a given factor, a new size, or
        a new voxel spacing. Order stands for interpolation order.

        Parameters:
        Image (SimpleITK.Image): The input image to be resampled.
        Factor (float, optional): The resampling factor. Defaults to None.
        Size (list, optional): The new size for the image. Defaults to [None].
        Spacing (list, optional): The new voxel spacing for the image. Defaults to [None].
        Order (int, optional): The interpolation order. Defaults to 1 (Linear interpolation).

        Returns:
        SimpleITK.Image: The resampled image.
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


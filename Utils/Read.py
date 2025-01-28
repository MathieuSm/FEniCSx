#%% !/usr/bin/env python3

"""
This module contains a class to read different types of files.
"""

__author__ = ['Mathieu Simon']
__date_created__ = '05-12-2024'
__license__ = 'MIT'
__version__ = '1.0'

import struct
import numpy as np

class Read():

    """
    Class to read different types of files.
    """

    def __init__(self):

        """
        Initialize the Read class.
        """
        
        pass

    def Fabric(self, FileName):

        """
        Read fabric data from an output file from medtool.

        Parameters:
        FileName (str): The name of the file to read.

        Returns:
        tuple: Eigenvalues, eigenvectors, and BVTV.
        """

        Text = open(FileName,'r').readlines()
        BVTV = float(Text[12].split('=')[1])
        eValues = np.array(Text[18].split(':')[1].split(),float)
        eVectors = np.zeros((3,3))
        for i in range(3):
            eVectors[i] = Text[19+i].split(':')[1].split()

        # Sort eigen values and eigen vectors
        Args = np.argsort(eValues)
        eValues = eValues[Args]
        eVectors = eVectors[Args]

        return eValues, eVectors, BVTV

    def ISQ(self, File, InfoFile=False, Echo=False, ASCII=False):

        """
        Read an ISQ file from Scanco and return a numpy array and additional data.

        Parameters:
        File (str): The name of the file to read.
        InfoFile (bool, optional): Whether to save the header information to a text file. Defaults to False.
        Echo (bool, optional): Whether to print the process information. Defaults to False.
        ASCII (bool, optional): Whether the file is in ASCII format. Defaults to False.

        Returns:
        tuple: Voxel model and additional data.
        """

        if Echo:
            Text = 'Read ISQ'
            Time.Process(1, Text)

        try:
            f = open(File, 'rb')
        except IOError:
            print("\n **ERROR**: ISQReader: intput file ' % s' not found!\n\n" % File)
            print('\n E N D E D  with ERRORS \n\n')

        for Index in np.arange(0, 200, 4):
            f.seek(Index)
            f.seek(Index)

        f.seek(32)
        CT_ID = struct.unpack('i', f.read(4))[0]

        f.seek(28)
        sample_nb = struct.unpack('i', f.read(4))[0]

        f.seek(108)
        Scanning_time = struct.unpack('i', f.read(4))[0] / 1000

        f.seek(168)
        Energy = struct.unpack('i', f.read(4))[0] / 1000.

        f.seek(172)
        Current = struct.unpack('i', f.read(4))[0]

        f.seek(44)
        X_pixel = struct.unpack('i', f.read(4))[0]

        f.seek(48)
        Y_pixel = struct.unpack('i', f.read(4))[0]

        f.seek(52)
        Z_pixel = struct.unpack('i', f.read(4))[0]

        f.seek(56)
        Res_General_X = struct.unpack('i', f.read(4))[0]

        f.seek(60)
        Res_General_Y = struct.unpack('i', f.read(4))[0]

        f.seek(64)
        Res_General_Z = struct.unpack('i', f.read(4))[0]

        Res_X = Res_General_X / float(X_pixel)
        Res_Y = Res_General_Y / float(Y_pixel)
        Res_Z = Res_General_Z / float(Z_pixel)

        Header_Txt = ['scanner ID:                 %s' % CT_ID,
                    'scaning time in ms:         %s' % Scanning_time,
                    'scaning time in ms:         %s' % Scanning_time,
                    'Energy in keV:              %s' % Energy,
                    'Current in muA:             %s' % Current,
                    'nb X pixel:                 %s' % X_pixel,
                    'nb Y pixel:                 %s' % Y_pixel,
                    'nb Z pixel:                 %s' % Z_pixel,
                    'resolution general X in mu: %s' % Res_General_X,
                    'resolution general Y in mu: %s' % Res_General_Y,
                    'resolution general Z in mu: %s' % Res_General_Z,
                    'pixel resolution X in mu:   %.2f' % Res_X,
                    'pixel resolution Y in mu:   %.2f' % Res_Y,
                    'pixel resolution Z in mu:   %.2f' % Res_Z]

        if InfoFile:
            Write_File = open(File.split('.')[0] + '_info.txt', 'w')
            for Item in Header_Txt:
                Write_File.write("%s\n" % Item)
            Write_File.close()

        f.seek(44)
        Header = np.zeros(6)
        for i in range(0, 6):
            Header[i] = struct.unpack('i', f.read(4))[0]

        ElementSpacing = [Header[3] / Header[0] / 1000, Header[4] / Header[1] / 1000, Header[5] / Header[2] / 1000]
        f.seek(508)

        HeaderSize = 512 * (1 + struct.unpack('i', f.read(4))[0])
        f.seek(HeaderSize)

        NDim = [int(Header[0]), int(Header[1]), int(Header[2])]
        LDim = [float(ElementSpacing[0]), float(ElementSpacing[1]), float(ElementSpacing[2])]

        AdditionalData = {'-LDim': LDim,
                        '-NDim': NDim,
                        'ElementSpacing': LDim,
                        'DimSize': NDim,
                        'HeaderSize': HeaderSize,
                        'TransformMatrix': [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        'CenterOfRotation': [0.0, 0.0, 0.0],
                        'Offset': [0.0, 0.0, 0.0],
                        'AnatomicalOrientation': 'LPS',
                        'ElementType': 'int16',
                        'ElementDataFile': File}

        if ASCII == False:
            VoxelModel = np.fromfile(f, dtype='i2')
            try:
                VoxelModel = VoxelModel.reshape((NDim[2], NDim[1], NDim[0]))
                f.close()
                del f

            except:
                # if the length does not fit the dimensions (len(VoxelModel) != NDim[2] * NDim[1] * NDim[0]),
                # add an offset with seek to reshape the image -> actualise length, delta *2 = seek

                Offset = (len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0]))
                f.seek(0)
                VoxelModel = np.fromfile(f, dtype='i2')

                if Echo:
                    print('len(VoxelModel) = ', len(VoxelModel))
                    print('Should be ', (NDim[2] * NDim[1] * NDim[0]))
                    print('Delta:', len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0]))

                f.seek((len(VoxelModel) - (NDim[2] * NDim[1] * NDim[0])) * 2)
                VoxelModel = np.fromfile(f, dtype='i2')
                f.close()
                del f

                VoxelModel = VoxelModel.reshape((NDim[2], NDim[1], NDim[0]))
                # the image is flipped by the Offset --> change the order to obtain the continuous image:
                VoxelModel = np.c_[VoxelModel[:, :, -Offset:], VoxelModel[:, :, :(VoxelModel.shape[2] - Offset)]]

        # If ISQ file was transfered with the Scanco microCT SFT in ASCII mode
        # a character is added every 256 bites, so build array by croping it
        else:
            LData = NDim[0] * NDim[1] * NDim[2]     # Data length
            NBytes = int(LData / 256)               # Number of bytes to store data
            Data = np.fromfile(f,dtype='i2')        # Read data
            Data = Data[::-1]                       # Reverse because data is at the end of the file
            cData = Data[:NBytes*257]               # Crop to data length
            sData = np.reshape(cData,(NBytes,257))  # Reshape for each 256 bytes
            sData = sData[:,1:]                     # Crop first byte artificially added by ascii ftp transfer
            
            # Reshape to scan dimensions
            VoxelModel = np.reshape(sData,(NDim[2],NDim[1],NDim[0]))

        if Echo:
            Time.Process(0,Text)
            print('\nScanner ID:                 ', CT_ID)
            print('Scanning time in ms:         ', Scanning_time)
            print('Energy in keV:              ', Energy)
            print('Current in muA:             ', Current)
            print('Nb X pixel:                 ', X_pixel)
            print('Nb Y pixel:                 ', Y_pixel)
            print('Nb Z pixel:                 ', Z_pixel)
            print('Pixel resolution X in mu:    %.2f' % Res_X)
            print('Pixel resolution Y in mu:    %.2f' % Res_Y)
            print('Pixel resolution Z in mu:    %.2f' % Res_Z)

        return VoxelModel, AdditionalData

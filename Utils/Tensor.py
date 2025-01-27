#%% !/usr/bin/env python3

Description = """
Class to perform tensor algebra
"""

__author__ = ['Mathieu Simon']
__date_created__ = '05-12-2024'
__date__ = '05-12-2024'
__license__ = 'MIT'
__version__ = '1.0'


import numpy as np
import pyvista as pv

class Tensor():

    def __init__(self):
        pass

 #%% Tensors products
    def Dyadic(self, A, B):

        if A.size == 3:

            C = np.zeros((3,3))

            for i in range(3):
                for j in range(3):
                    C[i,j] = A[i]*B[j]

        elif A.size == 9:

            C = np.zeros((3,3,3,3))

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            C[i,j,k,l] = A[i,j] * B[k,l]


        else:
            print('Matrices sizes mismatch')

        return C

    def Symmetric(self, A, B):

        C = np.zeros((3, 3, 3, 3))

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        C[i,j,k,l] = (1/2)*(A[i,k]*B[j,l]+A[i,l]*B[j,k])

        return C

    def Frobenius(self, A, B):

        s = 0

        if A.size == 9 and B.size == 9:
            for i in range(3):
                for j in range(3):
                    s += A[i, j] * B[i, j]

        elif A.size == 36 and B.size == 36:
            for i in range(6):
                for j in range(6):
                    s = s + A[i, j] * B[i, j]

        elif A.shape == (9,9) and B.shape == (9,9):
            for i in range(9):
                for j in range(9):
                    s = s + A[i, j] * B[i, j]

        elif A.shape == (3,3,3,3) and B.shape == (3,3,3,3):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            s = s + A[i, j, k, l] * B[i, j, k, l]

        else:
            print('Matrices sizes mismatch')

        return s

#%% Tensors manipulations
    def Engineering2MandelNotation(self, A):

        B = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                if i < 3 and j >= 3:
                    B[i,j] = A[i,j] * np.sqrt(2)
                elif i >= 3 and j < 3:
                    B[i,j] = A[i,j] * np.sqrt(2)
                elif i >= 3 and j >= 3:
                    B[i,j] = A[i,j] * 2
                else:
                    B[i, j] = A[i, j]

        return B

    def Mandel2EngineeringNotation(self, A):

        B = np.zeros((6,6))

        for i in range(6):
            for j in range(6):

                if i < 3 and j >= 3:
                    B[i,j] = A[i,j] / np.sqrt(2)

                elif i >= 3 and j < 3:
                    B[i,j] = A[i,j] / np.sqrt(2)

                elif i >= 3 and j >= 3:
                    B[i,j] = A[i,j] / 2

                else:
                    B[i, j] = A[i, j]

        return B

    def CheckMinorSymmetry(self, A):
        MinorSymmetry = True
        for i in range(3):
            for j in range(3):
                PartialTensor = A[:,:, i, j]
                if PartialTensor[1, 0] == PartialTensor[0, 1] and PartialTensor[2, 0] == PartialTensor[0, 2] and PartialTensor[1, 2] == PartialTensor[2, 1]:
                    MinorSymmetry = True
                else:
                    MinorSymmetry = False
                    break

        if MinorSymmetry == True:
            for i in range(3):
                for j in range(3):
                    PartialTensor = np.squeeze(A[i, j,:,:])
                    if PartialTensor[1, 0] == PartialTensor[0, 1] and PartialTensor[2, 0] == PartialTensor[0, 2] and PartialTensor[1, 2] == PartialTensor[2, 1]:
                        MinorSymmetry = True
                    else:
                        MinorSymmetry = False
                        break

        return MinorSymmetry

    def IsoMorphism66_3333(self, A):

        # Check symmetry
        Symmetry = True
        for i in range(6):
            for j in range(6):
                if not A[i,j] == A[j,i]:
                    Symmetry = False
                    break
        if Symmetry == False:
            print('Matrix is not symmetric!')
            return

        B = np.zeros((3,3,3,3))

        # Build 4th tensor
        B[0, 0, 0, 0] = A[0, 0]
        B[1, 1, 0, 0] = A[1, 0]
        B[2, 2, 0, 0] = A[2, 0]
        B[1, 2, 0, 0] = A[3, 0] / np.sqrt(2)
        B[2, 0, 0, 0] = A[4, 0] / np.sqrt(2)
        B[0, 1, 0, 0] = A[5, 0] / np.sqrt(2)

        B[0, 0, 1, 1] = A[0, 1]
        B[1, 1, 1, 1] = A[1, 1]
        B[2, 2, 1, 1] = A[2, 1]
        B[1, 2, 1, 1] = A[3, 1] / np.sqrt(2)
        B[2, 0, 2, 1] = A[4, 1] / np.sqrt(2)
        B[0, 1, 2, 1] = A[5, 1] / np.sqrt(2)

        B[0, 0, 2, 2] = A[0, 2]
        B[1, 1, 2, 2] = A[1, 2]
        B[2, 2, 2, 2] = A[2, 2]
        B[1, 2, 2, 2] = A[3, 2] / np.sqrt(2)
        B[2, 0, 2, 2] = A[4, 2] / np.sqrt(2)
        B[0, 1, 2, 2] = A[5, 2] / np.sqrt(2)

        B[0, 0, 1, 2] = A[0, 3] / np.sqrt(2)
        B[1, 1, 1, 2] = A[1, 3] / np.sqrt(2)
        B[2, 2, 1, 2] = A[2, 3] / np.sqrt(2)
        B[1, 2, 1, 2] = A[3, 3] / 2
        B[2, 0, 1, 2] = A[4, 3] / 2
        B[0, 1, 1, 2] = A[5, 3] / 2

        B[0, 0, 2, 0] = A[0, 4] / np.sqrt(2)
        B[1, 1, 2, 0] = A[1, 4] / np.sqrt(2)
        B[2, 2, 2, 0] = A[2, 4] / np.sqrt(2)
        B[1, 2, 2, 0] = A[3, 4] / 2
        B[2, 0, 2, 0] = A[4, 4] / 2
        B[0, 1, 2, 0] = A[5, 4] / 2

        B[0, 0, 0, 1] = A[0, 5] / np.sqrt(2)
        B[1, 1, 0, 1] = A[1, 5] / np.sqrt(2)
        B[2, 2, 0, 1] = A[2, 5] / np.sqrt(2)
        B[1, 2, 0, 1] = A[3, 5] / 2
        B[2, 0, 0, 1] = A[4, 5] / 2
        B[0, 1, 0, 1] = A[5, 5] / 2



        # Add minor symmetries ijkl = ijlk and ijkl = jikl

        B[0, 0, 0, 0] = B[0, 0, 0, 0]
        B[0, 0, 0, 0] = B[0, 0, 0, 0]

        B[0, 0, 1, 0] = B[0, 0, 0, 1]
        B[0, 0, 0, 1] = B[0, 0, 0, 1]

        B[0, 0, 1, 1] = B[0, 0, 1, 1]
        B[0, 0, 1, 1] = B[0, 0, 1, 1]

        B[0, 0, 2, 1] = B[0, 0, 1, 2]
        B[0, 0, 1, 2] = B[0, 0, 1, 2]

        B[0, 0, 2, 2] = B[0, 0, 2, 2]
        B[0, 0, 2, 2] = B[0, 0, 2, 2]

        B[0, 0, 0, 2] = B[0, 0, 2, 0]
        B[0, 0, 2, 0] = B[0, 0, 2, 0]



        B[0, 1, 0, 0] = B[0, 1, 0, 0]
        B[1, 0, 0, 0] = B[0, 1, 0, 0]

        B[0, 1, 1, 0] = B[0, 1, 0, 1]
        B[1, 0, 0, 1] = B[0, 1, 0, 1]

        B[0, 1, 1, 1] = B[0, 1, 1, 1]
        B[1, 0, 1, 1] = B[0, 1, 1, 1]

        B[0, 1, 2, 1] = B[0, 1, 1, 2]
        B[1, 0, 1, 2] = B[0, 1, 1, 2]

        B[0, 1, 2, 2] = B[0, 1, 2, 2]
        B[1, 0, 2, 2] = B[0, 1, 2, 2]

        B[0, 1, 0, 2] = B[0, 1, 2, 0]
        B[1, 0, 2, 0] = B[0, 1, 2, 0]



        B[1, 1, 0, 0] = B[1, 1, 0, 0]
        B[1, 1, 0, 0] = B[1, 1, 0, 0]

        B[1, 1, 1, 0] = B[1, 1, 0, 1]
        B[1, 1, 0, 1] = B[1, 1, 0, 1]

        B[1, 1, 1, 1] = B[1, 1, 1, 1]
        B[1, 1, 1, 1] = B[1, 1, 1, 1]

        B[1, 1, 2, 1] = B[1, 1, 1, 2]
        B[1, 1, 1, 2] = B[1, 1, 1, 2]

        B[1, 1, 2, 2] = B[1, 1, 2, 2]
        B[1, 1, 2, 2] = B[1, 1, 2, 2]

        B[1, 1, 0, 2] = B[1, 1, 2, 0]
        B[1, 1, 2, 0] = B[1, 1, 2, 0]



        B[1, 2, 0, 0] = B[1, 2, 0, 0]
        B[2, 1, 0, 0] = B[1, 2, 0, 0]

        B[1, 2, 1, 0] = B[1, 2, 0, 1]
        B[2, 1, 0, 1] = B[1, 2, 0, 1]

        B[1, 2, 1, 1] = B[1, 2, 1, 1]
        B[2, 1, 1, 1] = B[1, 2, 1, 1]

        B[1, 2, 2, 1] = B[1, 2, 1, 2]
        B[2, 1, 1, 2] = B[1, 2, 1, 2]

        B[1, 2, 2, 2] = B[1, 2, 2, 2]
        B[2, 1, 2, 2] = B[1, 2, 2, 2]

        B[1, 2, 0, 2] = B[1, 2, 2, 0]
        B[2, 1, 2, 0] = B[1, 2, 2, 0]



        B[2, 2, 0, 0] = B[2, 2, 0, 0]
        B[2, 2, 0, 0] = B[2, 2, 0, 0]

        B[2, 2, 1, 0] = B[2, 2, 0, 1]
        B[2, 2, 0, 1] = B[2, 2, 0, 1]

        B[2, 2, 1, 1] = B[2, 2, 1, 1]
        B[2, 2, 1, 1] = B[2, 2, 1, 1]

        B[2, 2, 2, 1] = B[2, 2, 1, 2]
        B[2, 2, 1, 2] = B[2, 2, 1, 2]

        B[2, 2, 2, 2] = B[2, 2, 2, 2]
        B[2, 2, 2, 2] = B[2, 2, 2, 2]

        B[2, 2, 0, 2] = B[2, 2, 2, 0]
        B[2, 2, 2, 0] = B[2, 2, 2, 0]



        B[2, 0, 0, 0] = B[2, 0, 0, 0]
        B[0, 2, 0, 0] = B[2, 0, 0, 0]

        B[2, 0, 1, 0] = B[2, 0, 0, 1]
        B[0, 2, 0, 1] = B[2, 0, 0, 1]

        B[2, 0, 1, 1] = B[2, 0, 1, 1]
        B[0, 2, 1, 1] = B[2, 0, 1, 1]

        B[2, 0, 2, 1] = B[2, 0, 1, 2]
        B[0, 2, 1, 2] = B[2, 0, 1, 2]

        B[2, 0, 2, 2] = B[2, 0, 2, 2]
        B[0, 2, 2, 2] = B[2, 0, 2, 2]

        B[2, 0, 0, 2] = B[2, 0, 2, 0]
        B[0, 2, 2, 0] = B[2, 0, 2, 0]


        # Complete minor symmetries
        B[0, 2, 1, 0] = B[0, 2, 0, 1]
        B[0, 2, 0, 2] = B[0, 2, 2, 0]
        B[0, 2, 2, 1] = B[0, 2, 1, 2]

        B[1, 0, 1, 0] = B[1, 0, 0, 1]
        B[1, 0, 0, 2] = B[1, 0, 2, 0]
        B[1, 0, 2, 1] = B[1, 0, 1, 2]

        B[2, 1, 1, 0] = B[2, 1, 0, 1]
        B[2, 1, 0, 2] = B[2, 1, 2, 0]
        B[2, 1, 2, 1] = B[2, 1, 1, 2]


        # Add major symmetries ijkl = klij
        B[0, 1, 1, 1] = B[1, 1, 0, 1]
        B[1, 0, 1, 1] = B[1, 1, 1, 0]

        B[0, 2, 1, 1] = B[1, 1, 0, 2]
        B[2, 0, 1, 1] = B[1, 1, 2, 0]


        return B

    def IsoMorphism3333_66(self, A):

        if self.CheckMinorSymmetry == False:
            print('Tensor does not present minor symmetry')
        else:

            B = np.zeros((6,6))

            B[0, 0] = A[0, 0, 0, 0]
            B[0, 1] = A[0, 0, 1, 1]
            B[0, 2] = A[0, 0, 2, 2]
            B[0, 3] = np.sqrt(2) * A[0, 0, 1, 2]
            B[0, 4] = np.sqrt(2) * A[0, 0, 2, 0]
            B[0, 5] = np.sqrt(2) * A[0, 0, 0, 1]

            B[1, 0] = A[1, 1, 0, 0]
            B[1, 1] = A[1, 1, 1, 1]
            B[1, 2] = A[1, 1, 2, 2]
            B[1, 3] = np.sqrt(2) * A[1, 1, 1, 2]
            B[1, 4] = np.sqrt(2) * A[1, 1, 2, 0]
            B[1, 5] = np.sqrt(2) * A[1, 1, 0, 1]

            B[2, 0] = A[2, 2, 0, 0]
            B[2, 1] = A[2, 2, 1, 1]
            B[2, 2] = A[2, 2, 2, 2]
            B[2, 3] = np.sqrt(2) * A[2, 2, 1, 2]
            B[2, 4] = np.sqrt(2) * A[2, 2, 2, 0]
            B[2, 5] = np.sqrt(2) * A[2, 2, 0, 1]

            B[3, 0] = np.sqrt(2) * A[1, 2, 0, 0]
            B[3, 1] = np.sqrt(2) * A[1, 2, 1, 1]
            B[3, 2] = np.sqrt(2) * A[1, 2, 2, 2]
            B[3, 3] = 2 * A[1, 2, 1, 2]
            B[3, 4] = 2 * A[1, 2, 2, 0]
            B[3, 5] = 2 * A[1, 2, 0, 1]

            B[4, 0] = np.sqrt(2) * A[2, 0, 0, 0]
            B[4, 1] = np.sqrt(2) * A[2, 0, 1, 1]
            B[4, 2] = np.sqrt(2) * A[2, 0, 2, 2]
            B[4, 3] = 2 * A[2, 0, 1, 2]
            B[4, 4] = 2 * A[2, 0, 2, 0]
            B[4, 5] = 2 * A[2, 0, 0, 1]

            B[5, 0] = np.sqrt(2) * A[0, 1, 0, 0]
            B[5, 1] = np.sqrt(2) * A[0, 1, 1, 1]
            B[5, 2] = np.sqrt(2) * A[0, 1, 2, 2]
            B[5, 3] = 2 * A[0, 1, 1, 2]
            B[5, 4] = 2 * A[0, 1, 2, 0]
            B[5, 5] = 2 * A[0, 1, 0, 1]

            return B
        
    def TransformTensor(self, A, OriginalBasis, NewBasis):

        # Build change of coordinate matrix
        O = OriginalBasis
        N = NewBasis

        Q = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                Q[i,j] = np.dot(O[i,:],N[j,:])

        if A.size == 36:
            A4 = self.IsoMorphism66_3333(A)

        elif A.size == 81 and A.shape == (3,3,3,3):
            A4 = A

        TransformedA = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            for n in range(3):
                                for o in range(3):
                                    for p in range(3):
                                        TransformedA[i, j, k, l] += Q[m,i]*Q[n,j]*Q[o,k]*Q[p,l] * A4[m, n, o, p]
        if A.size == 36:
            TransformedA = self.IsoMorphism3333_66(TransformedA)

        return TransformedA

    def Transform(self, A, B):

        if A.size == 9 and B.size == 3:

            c = np.zeros(3)
            for i in range(3):
                for j in range(3):
                    c[i] += A[i,j] * B[j]

            return c

        elif A.size == 27 and B.size == 9:

            c = np.zeros(3)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        c[i] += A[i,j,k] * B[j,k]

            return c

        elif A.size == 81 and B.size == 9:

            C = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            C[i,j] += A[i,j,k,l] * B[k,l]

            return C

        else:
            print('Matrices sizes mismatch')

#%% Tensors building
    def Fabric(self, eValues, eVectors):

        M = np.zeros((3,3))
        for m in range(3):
            M += eValues[m] * self.Dyadic(eVectors[m], eVectors[m])

        return M

    def Isotropic(self, E, Nu, Basis=np.eye(3)):

        """Build the full 3x3x3x3 isotropic elasticity tensor."""
        
        # Lamé parameters
        Lambda = E * Nu / ((1 + Nu) * (1 - 2 * Nu))
        Mu = E / (2 * (1 + Nu))

        # Build tensor
        Tensor = np.zeros((3, 3, 3, 3))
        for i in range(3):
            Mi = self.Dyadic(Basis[i], Basis[i])
            Tensor += (Lambda + 2*Mu) * self.Dyadic(Mi, Mi)
            for j in range(3):
                if i != j:
                    Mi = self.Dyadic(Basis[i], Basis[i])
                    Mj = self.Dyadic(Basis[j], Basis[j])
                    Tensor += Lambda * self.Dyadic(Mi, Mj)
                    Tensor += 2 * Mu * self.Symmetric(Mi, Mj)

        return Tensor

    def Othotropic(self, E1, E2, E3, Mu23, Mu31, Mu12, Nu23, Nu31, Nu12, Basis=np.eye(3)):
        
        """Build the full 3x3x3x3 transverse isotropic elasticity tensor."""

        # Lamé parameters
        Lambda1 = E1 * Nu23 / ((1 + Nu23) * (1 - 2 * Nu23))
        Lambda2 = E2 * Nu31 / ((1 + Nu31) * (1 - 2 * Nu31))
        Lambda3 = E3 * Nu12 / ((1 + Nu12) * (1 - 2 * Nu12))
        Lambdas = [Lambda1, Lambda2, Lambda3]
        Mus = [Mu23, Mu31, Mu12]

        # Build tensor
        Tensor = np.zeros((3, 3, 3, 3))
        for i in range(3):
            Mi = self.Dyadic(Basis[i], Basis[i])
            Tensor += (Lambdas[i] + 2*Mus[i]) * self.Dyadic(Mi, Mi)
            for j in range(3):
                if i != j:
                    Mi = self.Dyadic(Basis[i], Basis[i])
                    Mj = self.Dyadic(Basis[j], Basis[j])
                    Tensor += 1/2 * (Lambdas[i] + Lambdas[j]) * self.Dyadic(Mi, Mj)
                    Tensor += 2 * Mus[j] * self.Symmetric(Mi, Mj)

        return Tensor

#%% Tensors characteristics
    def Norm(self, A):
        return np.sqrt(np.sum(A**2))
    
#%% Tensors plotting
    def Ellipsoid(self, EigenValues, EigenVectors, NPoints=100):

        # New coordinate system
        Q = np.array(EigenVectors)

        # Build ellipsoid
        u = np.arange(0, 2 * np.pi + 2 * np.pi / NPoints, 2 * np.pi / NPoints)
        v = np.arange(0, np.pi + np.pi / NPoints, np.pi / NPoints)
        X = EigenValues[0] * np.outer(np.cos(u), np.sin(v))
        Y = EigenValues[1] * np.outer(np.sin(u), np.sin(v))
        Z = EigenValues[2] * np.outer(np.ones_like(u), np.cos(v))
        nNorm = np.zeros(X.shape)

        for i in range(len(X)):
            for j in range(len(X)):
                [X[i, j], Y[i, j], Z[i, j]] = np.dot([X[i, j], Y[i, j], Z[i, j]], Q)

        return X, Y, Z

    def ProjectEllipsoid(self, A, Normal):
        
        # Projection onto plane
        P = np.eye(3) - Tensor.Dyadic(self, Normal, Normal)
        B = P @ A @ P

        # Compute eigenvalues and vectors
        eVals, eVecs = np.linalg.eig(B)
        eVecs = eVecs.T

        # Keep only relevant values and vectors
        Indices = np.where(eVals != 0)[0]

        return eVals[Indices], eVecs[Indices]

    def PlotFabric(self, eValues:np.array, eVectors:np.array, FileName='', ROI=np.zeros((1,1,1))) -> None:

        """
        Plots a 3D ellipsoid representing a region of interest (ROI) with scaling based on the
        eigenvalues and eigenvectors provided. The ellipsoid is overlaid on a binary structure mesh,
        and the plot is generated with the ability to visualize the MIL (Mean Intercept Length) values.

        Parameters:
        -----------
        ROI (3D array): A 3D binary array representing the region of interest (ROI).
            
        eValues (1D array): A 1D array containing the eigenvalues of the fabric.
            
        eVectors (3D array) : A 2D array (shape: 3x3) containing the eigenvectors of the fabric.
            
        Returns:
        --------
        None
        """

        # Create a unit sphere and transform it to an ellipsoid
        Sphere = pv.Sphere(radius=ROI.shape[0]/2, theta_resolution=50, phi_resolution=50)

        # Scale the sphere by the square roots of the eigenvalues
        ScaleMatrix = np.diag(np.sqrt(eValues))
        TransformMatrix = np.matmul(eVectors, ScaleMatrix)

        # Transform the sphere points to ellipsoid points
        Points = np.matmul(Sphere.points, TransformMatrix.T)

        # Center the ellipsoid at the structure's midpoint
        Offset = np.array(ROI.shape) / 2
        EllispoidPoints = Points + Offset
        Ellispoid = pv.PolyData(EllispoidPoints, Sphere.faces)

        # Calculate the radius for each ellipsoid point to color by radius
        Radii = np.linalg.norm(Ellispoid.points - Offset, axis=1)
        Radii = (Radii - min(Radii)) / (max(Radii) - min(Radii))
        Radii = Radii * (max(eValues) - min(eValues)) + min(eValues)
        Ellispoid['MIL'] = Radii

        # Plotting
        sargs = dict(font_family='times', 
                        width=0.05,
                        height=0.75,
                        vertical=True,
                        position_x=0.9,
                        position_y=0.125,
                        title_font_size=30,
                        label_font_size=20
                        )
        
        pl = pv.Plotter(off_screen=True)
        if (ROI != 0).any():
            pl.add_mesh(ROI, cmap='bone', show_scalar_bar=False, opacity=0.5)
        pl.add_mesh(Ellispoid, scalars='MIL', cmap='jet', scalar_bar_args=sargs)
        pl.camera_position = 'xz'
        pl.camera.roll = 0
        pl.camera.elevation = 30
        pl.camera.azimuth = 30
        pl.camera.zoom(1.0)
        pl.add_bounding_box(color=(0,0,0), line_width=1)
        if len(FileName) > 0:
            pl.screenshot(FileName)
        else:
            pl.show()

        return

    def PlotTensor(self, Tensor:np.array, FileName='') -> None:

        """
        Plots a 3D ellipsoid representing a region of interest (ROI) with scaling based on the
        eigenvalues and eigenvectors provided. The ellipsoid is overlaid on a binary structure mesh,
        and the plot is generated with the ability to visualize the MIL (Mean Intercept Length) values.

        Parameters:
        -----------
        ROI (3D array): A 3D binary array representing the region of interest (ROI).
            
        eValues (1D array): A 1D array containing the eigenvalues of the fabric.
            
        eVectors (3D array) : A 2D array (shape: 3x3) containing the eigenvectors of the fabric.
            
        Returns:
        --------
        None
        """

        # Create a unit sphere and transform it to an ellipsoid
        Sphere = pv.Sphere(radius=1, theta_resolution=50, phi_resolution=50)

        # Compute elongation and bulk modulus
        I = np.eye(3)
        ElongationModulus = np.zeros(Sphere.points.shape)
        BulkModulus = np.zeros(len(Sphere.points))
        for p, Point in enumerate(Sphere.points):
            N = self.Dyadic(Point, Point)
            SN = self.Transform(Tensor, N)
            ElongationModulus[p] = self.Frobenius(N, SN) * Point
            BulkModulus[p] = self.Frobenius(I, SN)

        # # Scale the sphere by the square roots of the eigenvalues

        # # Center the ellipsoid at the structure's midpoint
        Ellispoid = pv.PolyData(ElongationModulus, Sphere.faces)
        Ellispoid['Bulk Modulus'] = BulkModulus

        # Plotting
        SArgs = dict(font_family='times', 
                    width=0.05,
                    height=0.75,
                    vertical=True,
                    position_x=0.85,
                    position_y=0.125,
                    title_font_size=30,
                    label_font_size=20
                    )
        
        # "all", "origin", "outer", "default", "closest", "front", "furthest", or "back"
        BArgs = dict(font_family='times', 
                    font_size=30,
                    location='default',
                    n_xlabels=1,
                    n_ylabels=1,
                    n_zlabels=1,
                    all_edges=True,
                    fmt='%i',
                    xtitle='',
                    ytitle='',
                    ztitle='',
                    use_3d_text=False
                    )
        
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(Ellispoid, scalars='Bulk Modulus', cmap='jet', scalar_bar_args=SArgs)
        if BulkModulus.max() / BulkModulus.min() < 1.01:
            pl.add_mesh(Ellispoid, color=(0.49, 1.0, 0.48))
        pl.camera_position = 'xz'
        pl.camera.roll = 0
        pl.camera.elevation = 30
        pl.camera.azimuth = 30
        pl.camera.zoom(0.8)
        pl.add_axes(viewport=(0,0,0.25,0.25),
                    label_size=(0.065, 0.065),
                    xlabel='e1',
                    ylabel='e2',
                    zlabel='e3')
        pl.show_bounds(**BArgs)

        if len(FileName) > 0:
            pl.screenshot(FileName, return_img=False, scale=2)
        else:
            pl.show()

        return


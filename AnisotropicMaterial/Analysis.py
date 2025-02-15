#%% #!/usr/bin/env python3

Description = """
This script performs a finite element analysis of an anisotropic material using the FEniCSx library. 
It reads a mesh from a .msh file, computes the volume of the mesh, defines material constants, 
builds compliance and stiffness matrices, and sets up the function space over the mesh domain. 
"""

__author__ = ['Mathieu Simon']
__date_created__ = '30-01-2025'
__license__ = 'GPL'
__version__ = '1.0'

#%% Imports

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats.distributions import t

sys.path.append(str(Path(__file__).parents[1]))
from Utils import Read

#%% Functions

def OLS(X, Y, Alpha=0.95, FName=''):

    """
    Performs Ordinary Least Squares (OLS) regression, computes confidence intervals for the parameters,
    and plots the regression results with different components of the stiffness tensor in different colors.
    
    Parameters:
    X (numpy.ndarray): The design matrix.
    Y (numpy.ndarray): The response vector.
    Alpha (float, optional): The confidence level for the confidence intervals. Default is 0.95.
    FName (str, optional): The file name to save the results. Default is an empty string.
    
    Returns:
    pandas.DataFrame: A DataFrame containing the parameter estimates and their confidence intervals.
    float: The adjusted R-squared value.
    numpy.ndarray: The norm error (NE) values.
    """

    # Solve linear system
    XTXi = np.linalg.inv(X.T * X)
    B = XTXi * X.T * Y

    # Compute residuals, variance, and covariance matrix
    Y_Obs = Y
    Y_Fit = X * B
    Residuals = Y - X*B
    DOFs = len(Y) - X.shape[1]
    Sigma = Residuals.T * Residuals / DOFs
    Cov = Sigma[0,0] * XTXi

    # Compute B confidence interval
    t_Alpha = t.interval(Alpha, DOFs)
    B_CI_Low = B.T + t_Alpha[0] * np.sqrt(np.diag(Cov))
    B_CI_Top = B.T + t_Alpha[1] * np.sqrt(np.diag(Cov))

    # Store parameters in data frame
    Parameters = pd.DataFrame(columns=[f'Parameter {i+1}' for i in range(len(B))])
    Parameters.loc['Value'] = [P[0] for P in np.array(B)]
    Parameters.loc['95% CI Low'] = [P for P in np.array(B_CI_Low)[0]]
    Parameters.loc['95% CI Top'] = [P for P in np.array(B_CI_Top)[0]]

    # Compute R2 and standard error of the estimate
    RSS = np.sum([R**2 for R in Residuals])
    SE = np.sqrt(RSS / DOFs)
    TSS = np.sum([R**2 for R in (Y - Y.mean())])
    RegSS = TSS - RSS
    R2 = RegSS / TSS

    # Compute R2adj and NE
    R2adj = 1 - RSS/TSS * (len(Y)-1)/(len(Y)-X.shape[1]-1)

    NE = []
    for i in range(0,len(Y),9):
        T_Obs = Y_Obs[i:i+12]
        T_Fit = Y_Fit[i:i+12]
        Numerator = np.sum([T**2 for T in (T_Obs-T_Fit)])
        Denominator = np.sum([T**2 for T in T_Obs])
        NE.append(np.sqrt(Numerator/Denominator))
    NE = np.array(NE)


    # Prepare data for plot
    Line = np.linspace(np.min(np.concatenate([X,Y])),
                       np.max(np.concatenate([X,Y])), len(Y))
    # B_0 = np.sort(np.sqrt(np.diag(X * Cov * X.T)))
    # CI_Line_u = np.exp(Line + t_Alpha[0] * B_0)
    # CI_Line_o = np.exp(Line + t_Alpha[1] * B_0)

    # Plots
    DPI = 500
    Colors=[(0,0,1),(0,1,0),(1,0,0)]

    # Elements
    ii = np.tile([1,0,0,1,0,1,0,0,0],len(X)//9).astype(bool)
    ij = np.tile([0,1,1,0,1,0,0,0,0],len(X)//9).astype(bool)
    jj = np.tile([0,0,0,0,0,0,1,1,1],len(X)//9).astype(bool)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=DPI)
    # Axes.fill_between(np.exp(Line), CI_Line_u, CI_Line_o, color=(0.8,0.8,0.8))
    Axes.plot(X[ii, 0], Y_Obs[ii],
              color=Colors[0], linestyle='none', marker='s')
    Axes.plot(X[ij, 0], Y_Obs[ij],
              color=Colors[1], linestyle='none', marker='o')
    Axes.plot(X[jj, 0], Y_Obs[jj],
              color=Colors[2], linestyle='none', marker='^')
    Axes.plot([], color=Colors[0], linestyle='none', marker='s', label=r'$\lambda_{ii}$')
    Axes.plot([], color=Colors[1], linestyle='none', marker='o', label=r'$\lambda_{ij}$')
    Axes.plot([], color=Colors[2], linestyle='none', marker='^', label=r'$\mu_{ij}$')
    Axes.plot(Line, Line, color=(0, 0, 0), linestyle='--')
    Axes.annotate(r'N ROIs   : ' + str(len(Y)//9), xy=(0.3, 0.1), xycoords='axes fraction')
    Axes.annotate(r'N Points : ' + str(len(Y)), xy=(0.3, 0.025), xycoords='axes fraction')
    Axes.annotate(r'$R^2_{ajd}$: ' + format(round(R2adj, 3),'.3f'), xy=(0.65, 0.1), xycoords='axes fraction')
    Axes.annotate(r'NE : ' + format(round(NE.mean(), 2), '.2f') + r'$\pm$' + format(round(NE.std(), 2), '.2f'), xy=(0.65, 0.025), xycoords='axes fraction')
    Axes.set_xlabel(r'Abaqus $\mathrm{\mathbb{S}}$ (GPa)')
    Axes.set_ylabel(r'FEniCS $\mathrm{\mathbb{S}}$ (GPa)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    if len(FName) > 0:
        plt.savefig(FName)
    plt.close(Figure)

    return Parameters, R2adj, NE


#%% Main

def Main():

    # Define paths
    AbaqusPath = Path(__file__).parent / 'Abaqus'
    FEniCSPath = Path(__file__).parent / 'FEniCS'

    # Read files
    Files = [F.stem for F in FEniCSPath.iterdir()]

    # Homogenization strains
    Strain = np.array([0.001, 0.001, 0.001, 0.002, 0.002, 0.002])

    FEniCS, Abaqus = [], [],
    for F in Files:

        # FEniCS stiffness
        Data = np.load(FEniCSPath / (F + '.npy'))
        FEniCS.append(1/2 * (Data + Data.T))

        # Abaqus stiffness
        Data = open(AbaqusPath / (F + '.out'), 'r').readlines()
        Stress = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                Stress[i,j] = float(Data[i+4].split()[j+1])

        # Compute stiffness
        A_Stiffness = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                A_Stiffness[i,j] = Stress[i,j] / Strain[i]
        Abaqus.append(1/2 * (A_Stiffness + A_Stiffness.T))

    # Build linear system
    X = np.matrix(np.ones((len(FEniCS)*9, 1)))
    Y = np.matrix(np.zeros((len(FEniCS)*9, 1)))
    for f in range(len(FEniCS)):
        
        Start, Stop = 9*f, 9*(f+1)
        X[Start:Stop] = [[Abaqus[f][0,0]],
                         [Abaqus[f][0,1]],
                         [Abaqus[f][0,2]],
                         [Abaqus[f][1,1]],
                         [Abaqus[f][1,2]],
                         [Abaqus[f][2,2]],
                         [Abaqus[f][3,3]],
                         [Abaqus[f][4,4]],
                         [Abaqus[f][5,5]]]
        
        Y[Start:Stop] = [[FEniCS[f][0,0]],
                         [FEniCS[f][0,1]],
                         [FEniCS[f][0,2]],
                         [FEniCS[f][1,1]],
                         [FEniCS[f][1,2]],
                         [FEniCS[f][2,2]],
                         [FEniCS[f][3,3]],
                         [FEniCS[f][4,4]],
                         [FEniCS[f][5,5]]]

    FName = Path(__file__).parent / 'Regression.png'
    Parameters, R2adj, NE = OLS(X, Y, FName=str(FName))



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

        
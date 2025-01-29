#%% !/usr/bin/env python3

Description = """
Read ISQ files and plot them in 3D using pyvista
"""

__author__ = ['Mathieu Simon']
__date_created__ = '12-11-2024'
__date__ = '15-11-2024'
__license__ = 'GPL'
__version__ = '1.0'


#%% Imports

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind
from scipy.stats.distributions import norm, t

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

    # Homogenization strains
    Strain = np.array([0.001, 0.001, 0.001, 0.002, 0.002, 0.002])

    # Build dataframe to store results
    FEniCSData = []
    AbaqusData = []

    # Define paths
    AbaqusPath = Path(__file__).parent / 'Abaqus'
    FEniCSPath = Path(__file__).parent / 'FEniCS'

    # List results
    FEniCS = sorted([F for F in FEniCSPath.iterdir() if F.name.endswith('.npy')])

    for i, F in enumerate(FEniCS):

        # FEniCS stiffness
        F_Stiffness = np.load(F)
        FEniCSData.append(1/2 * (F_Stiffness + F_Stiffness.T))

        # Abaqus stiffness
        File = open(AbaqusPath / (F.name[:-4] + '.out'), 'r').readlines()
        Stress = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                Stress[i,j] = float(File[i+4].split()[j+1])

        # Compute stiffness
        A_Stiffness = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                A_Stiffness[i,j] = Stress[i,j] / Strain[i]

        # Symetrize matrix
        AbaqusData.append(1/2 * (A_Stiffness + A_Stiffness.T))
        
    # Build linear system
    X = np.matrix(np.ones((len(FEniCSData)*9, 1)))
    Y = np.matrix(np.zeros((len(FEniCSData)*9, 1)))
    for f in range(len(FEniCSData)):
        
        Start, Stop = 9*f, 9*(f+1)
        X[Start:Stop] = [[AbaqusData[f][0,0]],
                         [AbaqusData[f][0,1]],
                         [AbaqusData[f][0,2]],
                         [AbaqusData[f][1,1]],
                         [AbaqusData[f][1,2]],
                         [AbaqusData[f][2,2]],
                         [AbaqusData[f][3,3]],
                         [AbaqusData[f][4,4]],
                         [AbaqusData[f][5,5]]]
        
        Y[Start:Stop] = [[FEniCSData[f][0,0]],
                         [FEniCSData[f][0,1]],
                         [FEniCSData[f][0,2]],
                         [FEniCSData[f][1,1]],
                         [FEniCSData[f][1,2]],
                         [FEniCSData[f][2,2]],
                         [FEniCSData[f][3,3]],
                         [FEniCSData[f][4,4]],
                         [FEniCSData[f][5,5]]]

    FName = Path(__file__).parent / 'Regression.png'
    Parameters, R2adj, NE = OLS(X, Y, FName=str(FName))


if __name__ == '__main__':
    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add optional argument
    ScriptVersion = Parser.prog + ' version ' + __version__
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()
    Main()

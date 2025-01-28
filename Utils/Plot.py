#%% !/usr/bin/env python3

"""
This module contains a class for producing standard plots using matplotlib, numpy, and pandas.
"""

__author__ = ['Mathieu Simon']
__date_created__ = '05-12-2024'
__license__ = 'MIT'
__version__ = '1.0'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.distributions import t


class Plot():
    
    """
    Class for producing standard plots.
    """
    
    def __init__(self):
        
        """
        Initialize the Plot class.
        """
        
        pass

    def OLS(self, X, Y, Alpha=0.95, CI_Area=True, XLabel='X Data', YLabel='Y Data', FileName='', Show=True):
        
        """
        Perform Ordinary Least Squares (OLS) regression and plot the results.

        Parameters:
        X (array-like): The independent variable data.
        Y (array-like): The dependent variable data.
        Alpha (float, optional): The confidence level for the confidence intervals. Defaults to 0.95.
        CI_Area (bool, optional): Whether to display the confidence interval area. Defaults to True.
        XLabel (str, optional): Label for the X-axis. Defaults to 'X Data'.
        YLabel (str, optional): Label for the Y-axis. Defaults to 'Y Data'.
        FileName (str, optional): Name of the file to save the plot. Defaults to ''.
        Show (bool, optional): Whether to show the plot. Defaults to True.

        Returns:
        tuple: DataFrame of OLS parameters, R-squared value, and standard error.
        """
        
        # Convert to matrices
        X = np.matrix(X).T
        Y = np.matrix(Y).T

        # Solve linear system
        XTXi = np.linalg.inv(X.T * X)
        B = XTXi * X.T * Y

        # Compute residuals, variance, and covariance matrix
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
        Columns = [f'Variable {V+1}' for V in range(len(B))]
        Parameters = pd.DataFrame(columns=Columns)
        Parameters.loc['Value'] = np.array(B)[:,0]
        Parameters.loc['95% CI Low'] = np.array(B_CI_Low)[0]
        Parameters.loc['95% CI Top'] = np.array(B_CI_Top)[0]

        # Compute R2 and standard error of the estimate
        RSS = np.sum([R**2 for R in Residuals])
        SE = np.sqrt(RSS / DOFs)
        TSS = np.sum([R**2 for R in (Y - Y.mean())])
        RegSS = TSS - RSS
        R2 = RegSS / TSS

        # Prepare data for plot
        Line = np.linspace(min(Y.min(), (X*B).min()), max(Y.max(), (X*B).max()), len(Y))
        B_0 = np.sort(np.sqrt(np.diag(X * Cov * X.T)))
        CI_Line_u = Line + t_Alpha[0] * B_0
        CI_Line_o = Line + t_Alpha[1] * B_0

        # Plots
        Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5))#, dpi=self.DPI)
        Axes.fill_between(Line, CI_Line_u, CI_Line_o, color=(0.8,0.8,0.8))
        Axes.plot(Line, Line, color=(0,0,0), linestyle='--')
        Axes.plot(Y, Y_Fit, color=(0,0,1), linestyle='none', marker='o', fillstyle='none')
        Axes.plot(Y_Fit, Y_Fit, color=(1,0,0))
        Axes.annotate(r'N Points : ' + str(len(Y)), xy=(0.7, 0.14), xycoords='axes fraction')
        Axes.annotate(r'$R^2$: ' + format(round(R2, 3),'.3f'), xy=(0.7, 0.08), xycoords='axes fraction')
        Axes.annotate(r'SE : ' + format(round(SE.mean(), 2), '.2f') + '$\pm$' + format(round(SE.std(), 2), '.2f'), xy=(0.7, 0.025), xycoords='axes fraction')
        Axes.set_xlabel(XLabel)
        Axes.set_ylabel(YLabel)
        if len(FileName) > 0:
            plt.savefig(FileName)
        if Show:
            plt.show()
        else:
            plt.close(Figure)

        return Parameters, R2, SE


#%% !/usr/bin/env python3

"""
This module contains a class to display a progress bar and measure the time to execute a process.
"""

__author__ = ['Mathieu Simon']
__date_created__ = '05-12-2024'
__license__ = 'MIT'
__version__ = '1.0'


import time
import numpy as np

class Time():
    
    """
    Class to display a progress bar and measure the time to execute a process.
    """

    def __init__(self):
        
        """
        Initialize the Time class with default width, length, text, and start time.
        """
        
        self.Width = 15
        self.Length = 16
        self.Text = 'Process'
        self.Tic = time.time()
    
    def Set(self, Tic=None):
        
        """
        Set the start time (Tic) for the process.

        Parameters:
        Tic (float, optional): Custom start time. Defaults to the current time.
        """
        
        if Tic is None:
            self.Tic = time.time()
        else:
            self.Tic = Tic

    def Print(self, Tic=None, Toc=None):
        
        """
        Print elapsed time in seconds to time in HH:MM:SS format.

        Parameters:
        Tic (float, optional): Start time of the process. Defaults to the stored start time.
        Toc (float, optional): End time of the process. Defaults to the current time.
        """
        
        if Tic is None:
            Tic = self.Tic
            
        if Toc is None:
            Toc = time.time()

        Delta = Toc - Tic

        Hours = np.floor(Delta / 60 / 60)
        Minutes = np.floor(Delta / 60) - 60 * Hours
        Seconds = Delta - 60 * Minutes - 60 * 60 * Hours

        print('\nProcess executed in %02i:%02i:%02i (HH:MM:SS)' % (Hours, Minutes, Seconds))

    def Update(self, Progress, Text=''):
        
        """
        Update the progress bar display.

        Parameters:
        Progress (float): Progress value between 0 and 1.
        Text (str, optional): Custom text for the progress bar. Defaults to the stored text.
        """
        
        Percent = int(round(Progress * 100))
        Np = self.Width * Percent // 100
        Nb = self.Width - Np

        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        Ns = self.Length - len(Text)
        if Ns >= 0:
            Text += Ns * ' '
        else:
            Text = Text[:self.Length]
        
        Line = '\r' + Text + ' [' + Np * '=' + Nb * ' ' + ']' + f' {Percent:.0f}%'
        print(Line, sep='', end='', flush=True)

    def Process(self, StartStop, Text=''):
        
        """
        Start or stop the process timer and update the progress bar.

        Parameters:
        StartStop (bool): Flag to start (True) or stop (False) the process timer.
        Text (str, optional): Custom text for the progress bar. Defaults to the stored text.
        """
        
        if len(Text) == 0:
            Text = self.Text
        else:
            self.Text = Text

        if StartStop:
            self.Tic = time.time()
            self.Update(0, Text)
        else:
            self.Update(1, Text)
            self.Print()

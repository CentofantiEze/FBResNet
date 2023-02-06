"""
SolverCheybyshev classe.

@author: Cecile Della Valle
@date: 03/01/2021
"""

# general
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.linalg import inv,pinvh,eig,eigh
from scipy.special import gamma
import pandas as pd
from scipy.interpolate import interp1d
# local
from FBRN.myfunc import Physics
from torch.autograd import Variable

class SolverChebyshev():
    def __init__(self,nx=2000, m=50, a=1, p=1):
        self.nx  = nx
        self.m   = m
        self.dx  = 1/(self.nx)
        self.p   = p
        self.a   = a
        # Matrice opérateur
        self.physics = Physics(self.nx, self.m, self.a, self.p)
        
    def Fourier_filter(self, data_loader,cut=0.5,display=False,idx=0):
        
        n_cut = int(self.m*cut)
        filter = np.zeros((self.m,self.m))
        np.fill_diagonal(filter[:n_cut,:n_cut], 1)
        
        # Extract data
        data = data_loader.dataset[:]
        y, x = np.squeeze(np.array(data[0])), np.squeeze(np.array(data[1]))
        
        # Iterate over the dataset
        err = 0
        err_list = []

        for (x_true, x_bias) in zip(x,y):
            # Recover the signal
            x_recovered = self.physics.inv.dot(x_bias)
            # Filter the signal
            x_filtered = filter.dot(x_recovered)
            # Change to the continuous functions basis
            x_filtered_elt = self.physics.BasisChangeInv(x_filtered)
            x_true_elt = self.physics.BasisChangeInv(x_true)
            # Compute the error
            err += np.sum((x_true_elt-x_filtered_elt)**2)/np.sum(x_true_elt**2)
            err_list.append(np.sum((x_true_elt-x_filtered_elt)**2)/np.sum(x_true_elt**2))
        # plot
        if display:
            x_bias=y[idx]
            x_true=x[idx]
            # Recover the signal
            x_recovered = self.physics.inv.dot(x_bias)
            # Filter the signal
            x_filtered = filter.dot(x_recovered)
            # Change to the continuous functions basis
            x_filtered_elt = self.physics.BasisChangeInv(x_filtered)
            x_true_elt = self.physics.BasisChangeInv(x_true)
            plt.plot(x_true_elt, label='true signal')
            plt.plot(x_filtered_elt, label='filtered signal')
            plt.legend()
            plt.show()
        return err/len(x), err_list

            
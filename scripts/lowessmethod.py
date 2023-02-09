"""
Lowess filter class.

@author: Ezequiel Centofanti
@date: 00/02/2023
"""

# general
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
# local
from FBRN.myfunc import Physics

class LowessSolver():
    def __init__(self,nx=2000, m=50, a=1, p=1):
        self.nx  = nx
        self.m   = m
        self.dx  = 1/(self.nx)
        self.p   = p
        self.a   = a
        # Matrice opérateur
        self.physics = Physics(self.nx, self.m, self.a, self.p)
        
    def Lowess_filter(self, data_loader,frac=200/2000,it=1,delta=50,project=True,display=False,idx=0):
        # Extract data
        data = data_loader.dataset[:]
        y, x = np.squeeze(np.array(data[0])), np.squeeze(np.array(data[1]))
        
        # Iterate over the dataset
        err = 0
        err_list = []

        for (x_true, x_bias) in zip(x,y):

            # Recover the signal
            x_recovered = self.physics.inv.dot(x_bias)
            x_recover_noisy_elt = self.physics.BasisChangeInv(x_recovered)
            # Filter the signal
            filtered = lowess(x_recover_noisy_elt, np.linspace(0,self.nx-1,self.nx),
                frac=frac,
                it=it,
                delta=delta)
            
            if project:
                # Project signal into eigenfunction space
                x_recovered_lowess = self.physics.BasisChangeInv(self.physics.BasisChange(filtered[:, 1]))
            else:
                x_recovered_lowess =filtered[:, 1]

            x_true_elt = self.physics.BasisChangeInv(x_true)
            # Compute the error
            err += np.sum((x_true_elt-x_recovered_lowess)**2)/np.sum(x_true_elt**2)
            err_list.append(np.sum((x_true_elt-x_recovered_lowess)**2)/np.sum(x_true_elt**2))
        # plot
        if display:
            x_bias=y[idx]
            x_true=x[idx]
            # Recover the signal
            x_recovered = self.physics.inv.dot(x_bias)
            x_recover_noisy_elt = self.physics.BasisChangeInv(x_recovered)
            # Filter the signal
            filtered = lowess(x_recover_noisy_elt, np.linspace(0,self.nx-1,self.nx),
                frac=frac,
                it=it,
                delta=delta)
            if project:
                # Project signal into eigenfunction space
                x_recovered_lowess = self.physics.BasisChangeInv(self.physics.BasisChange(filtered[:, 1]))
            else:
                x_recovered_lowess =filtered[:, 1]
                
            x_true_elt = self.physics.BasisChangeInv(x_true)
            plt.plot(x_true_elt, label='true signal')
            plt.plot(x_recovered_lowess, label='filtered signal')
            plt.legend()
            plt.show()
        return err/len(x), err_list

        
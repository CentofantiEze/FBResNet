"""
Classes and Functions used in the FBResNet model.
Classes
-------
    Physics  : Define the physical parameters of the ill-posed problem.
    MyMatmul : Multiplication with a kernel (for single or batch)
Methods
-------
    Export_Data : save a signal or function x 
    Export_hyper : hyperparameters of the neural network
     
@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
import numpy.linalg as la
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`

#

#
class Physics:
    """
    Define the physical parameters of the ill-posed problem.
    Alert : nx must be >> than m.
    Attributes
    ----------
        nx          (int): size of initial signal 
        m           (int): size of eigenvectors span
        a           (int): oder of ill-posedness 
        p           (int): order of a priori smoothness
        eigm   (np.array): transformation between signal and eigenvectors basis
        Top    (np.array): Abel operator from the finite element basis to the cos basis
        Ta     (np.array): T operator in the finite element basis (discretized integral)
        Tadj   (np.array): Adjoint of the T operator in the finite element basis.
        IP_mat (np.array): Matrix for the inner product discretization.
    """
    def __init__(self,nx=2000,m=50,a=1,p=1,discrete_op=True):
        """
        Alert : nx must be >> than m.

        Parameters
        ----------
            nx           (int): size of initial signal 
            m            (int): size of eigenvectors span
            a            (int): oder of ill-posedness 
            p            (int): order of a priori smoothness
        """
        # Physical parameters
        self.nx   = nx
        self.m    = m
        self.a    = a
        self.p    = p
        # Discrete inner product matrix
        kernel = 2*np.ones(self.nx)
        kernel[0] = 1
        self.IP_mat=np.diag(kernel)
        # Discretization stepsize     
        h          = 1/(self.nx)
        # Gamma(a)
        g = gamma_func(self.a)

        # T and T adjoint operators in the finite elements basis
        Ta = np.zeros((self.nx,self.nx))
        for i in range(self.nx):
            for j in range(self.nx):
                if j<i:
                    Ta[i,j]=h**self.a/(2*g*self.a) * ((i-j+1)**self.a-(i-j-1)**self.a)
                if j==0 and i!=0:
                    Ta[i,j]=h**self.a/(2*g*self.a) * ((i)**self.a - (i-1)**self.a)
                if j==i and i!=0:
                    Ta[i,j]=h**self.a/(2*g*self.a)
                if i==j and i==0:
                    Ta[i,j]=0
                if j>i:
                    Ta[i,j]=0
        self.Ta = Ta

        Tadj = np.zeros((self.nx,self.nx))
        for i in range(self.nx):
            for j in range(self.nx):
                if j>i and j!=(self.nx-1):
                    Tadj[i,j]=h**(self.a)/(2*g*(self.a)) * ((j-i+1)**(self.a)-(j-i-1)**(self.a))
                if j==(self.nx-1) and i!=j:
                    Tadj[i,j]=h**(self.a)/(2*g*(self.a)) * (2* (j-i+1)**(self.a) - (j-i)**(self.a) - (j-i-1)**(self.a))
                if j==i and i!=(self.nx-1):
                    Tadj[i,j]=h**(self.a)/(2*g*(self.a))
                if i==j and i==(self.nx-1):
                    Tadj[i,j]=h**(self.a)/(g*(self.a))
                if j<i:
                    Tadj[i,j]=0
        self.Tadj = Tadj

        
        
        # Get eigenvectors and eigenvalues of T*T
        eigw_full, base_full = la.eig(self.Tadj.dot(self.Ta))
        # Keep the first 'm' eigenvectors
        base_m = base_full[:,:self.m].T
        # Set sign of eigv, s.t. v[0]>0
        base = np.diag(np.sign(base_m[:,0])).dot(base_m)
        base[:,0] = base[0,0]
        # Normalize the base
        base_norm = self.normalize_base(base)
        self.basis = base_norm
        # Eigenvalues
        self.eigw = eigw_full[:self.m]
        # Inverse operator of T*T in the eigenbasis
        self.inv      = np.diag(self.eigw**(-1))
        self.eigm     = self.eigw**(-1/(2*self.a))
        

    def inner_prod(self,f1,f2):
        return f1.T.dot(self.IP_mat).dot(f2)/(2*self.nx)

    def normalize_base(self, base):
        for i, f in enumerate(base):
            base[i] = f/np.sqrt(self.inner_prod(f,f))
        return base

    def project(self, x):
        return (self.basis).dot((self.IP_mat).dot(np.squeeze(x)))/(2*self.nx)

    def BasisChange(self,x):
        """
        Change basis from signal to eigenvectors span.
        Parameters
        ----------
            x (np.array): signal of size n*c*nx
        Returns
        -------
            (np.array): of size n*c*m
        """
        return self.project(x)

    def BasisChangeInv(self,x):
        """
        Change basis from eigenvectors span to signal.
        Parameters
        ----------
            x (np.array): signal of size nxcxm
        Returns
        -------
            (np.array): of size nxcxnx
        """
        return np.matmul(x,self.basis)
    
    def Operators(self):
       """
       Given a ill-posed problem of order a and an a priori of order p
       for a 1D signal of nx points,
       the fonction computes the array of the linear transformation T
       and arrays used in the algorithm.
       Returns
       -------
           (list): four numpy array, the regularisation a priori, the Abel operator,
                   the ortogonal matrix from element to eigenvector basis
       """
       Top      = np.diag(1/self.eigm**(self.a))
       Dop      = np.diag(self.eigm**(self.p))
       # matrix P: basis change from cos <-> elt
       eltTocos = (self.basis).dot(self.IP_mat)/(2*self.nx)
       cosToelt = self.basis.T
       # TT and DD operators in eig space
       tDD      = Dop*Dop
       tTT      = Top*Top
       #
       return [tDD,tTT,eltTocos,cosToelt]
      
    def Compute(self,x):
        """
        Compute the transformation by the Abel integral operator
        in the basis of finite element.
        Parameters
        ----------
            x (np.array): signal of size n*c*nx
        Returns
        -------
            (np.array): of size n*c*nx
        """
        return np.matmul(x, self.Ta.T)

    
    def ComputeAdjoint(self,y):
        """
        Compute the transformation by the adjoint operator of Abel integral
        from the basis of finite element to eigenvectors.
        Parameters
        ----------
            x (np.array): signal of size n*c*nx
        Returns
        -------
            (np.array): of size n*c*m
        """
        return self.BasisChange(np.matmul(y, self.Tadj.T))

#
class MyMatmul(nn.Module):
    """
    Performs 1D convolution with numpy array kernel
    Attributes
    ----------
        kernel (torch.FloatTensor): size nx*nx filter
    """
    def __init__(self, kernel):
        """
        Parameters
        ----------
            kernel (numpy array): convolution filter
        """
        super(MyMatmul, self).__init__()
        kernel_nn     = torch.FloatTensor(kernel)
        self.kernel   = nn.Parameter(kernel_nn.T,requires_grad=False)   
            
    def forward(self, x): 
        """
        Performs convolution.
        Parameters
        ----------
            x (torch.FloatTensor): 1D-signal, size n*c*nx
        Returns
        -------
            (torch.FloatTensor): result of the convolution, size n*c*nx
        """
        x_tilde = torch.matmul(x,self.kernel)
        return x_tilde


####################################################################
####################################################################

### EXPORT DATA
def Export_Data(xdata,ydata,folder,name,header=True):
    """
    Save a signal in a chose folder
    for plot purpose.
    """
    Npoint = np.size(xdata)
    with open(folder+'/'+name+'.txt', 'w') as f:
        if header:
            f.writelines('xdata ydata \n')
        for i in range(Npoint):
            web_browsers = ['{0}'.format(xdata[i]),' ','{0} \n'.format(ydata[i])]
            f.writelines(web_browsers)

### PLOT GAMMA ALPHA MU
def Export_hyper(resnet,x,x_b,folder):
    """
    Export hyperparameters of a neural network
    """
    nlayer = len(resnet.model.Layers)
    gamma  = np.zeros(nlayer)
    reg    = np.zeros(nlayer)
    mu     = np.zeros(nlayer)
    for i in range(0,nlayer):
        gamma[i] = resnet.model.Layers[i].gamma_reg[0]
        reg[i]   = resnet.model.Layers[i].gamma_reg[1]
        mu[i]    = resnet.model.Layers[i].mu
    # export
    num    = np.linspace(0,nlayer-1,nlayer)
    Export_Data(num, gamma, folder, 'gradstep')
    Export_Data(num, reg, folder, 'reg')
    Export_Data(num, mu, folder, 'prox')
    # plot
    fig, (ax0,ax1,ax2) = plt.subplots(1, 3)
    ax0.plot(num,gamma)
    ax0.set_title('gradstep')
    ax1.plot(num,reg)
    ax1.set_title('reg')
    ax2.plot(num,mu)
    ax2.set_title('prox')
    plt.show()

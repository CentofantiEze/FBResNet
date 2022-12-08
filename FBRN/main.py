"""
FBRestNet class.
FBRN
Classes
-------
    FBResNet      : Forward Backward Residual Neural Network 
                    Training, Validation, Robustness
-------
Il want to thank Marie-Caroline Corbineau
https://github.com/mccorbineau/iRestNet
from which work this code and architecture is largely inspired.
-------

@author: Cecile Della Valle
@date: 03/01/2021
"""

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# General import
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from torch.autograd import Variable
import numpy as np
import pandas as pd
import cv2 as cv
import os
from PIL import Image
import matplotlib.pyplot as plt
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# Local import
from FBRN.myfunc import Physics
from FBRN.myfunc import MyMatmul
from FBRN.model import MyModel
from FBRN.myfunc import Export_Data

        
class FBRestNet(nn.Module):
    """
    Includes the main training and testing methods of iRestNet.
    Attributes
    ----------
        model_id             (str): the unique Id of the model.
        physics    (Physics class): contains the parameters of the Abel integral.
        noise              (float): standard deviation of the Gaussian white noise.
        constr               (str): 'cube' of 'slab' to determine the proximal operator.
        lr_i               (float): learning rate.
        nb_epochs            (int): number of epochs of training.
        freq_val             (int): Model validation rate.
        nb_blocks            (int): number of blocks in the neural network.
        batch_size           (int): training batch size.
        train_size           (int): size of the training dataset.
        val_size             (int): size of the validation dataset.
        im_set               (str): images used to construct the 1D signal dataset (Set1 or Set2).
        loss_fn             (Loss): loss function (here only MSE).
        loss_elt            (bool): compute the loss in the finite elements space.
        dataset_folder       (str): path to the Dataset folder.
        model_folder         (str): path to the pretrained model weigths.
        opt_hist_folder      (str): optimisation history saving folder.
        save_signals        (bool): save the 1D signals.
        save_outputs        (bool): save the forwar-backwards parameters.
        save_model          (bool): save the models weights.
        save_hist           (bool): save the models optimisation history.
        regul               (bool): 'True' is the regularisation parameter is not zero.
        model      (Mymodel class): the FBResNet model for Abel integral inversion.
    """
#========================================================================================================
#========================================================================================================
    def __init__(
        self,
        model_id = 'model_000_',
        dataset_folder = '../Datasets/',
        model_folder = '../outputs/models/',
        opt_hist_folder = '../outputs/opt_hist/',
        experimentation=Physics(2000,50,1,1),
        nb_blocks=20,
        im_set="Set1",
        noise = 0.05,        
        constraint = 'cube',  
        train_size=50,
        val_size=10,
        batch_size=5,
        lr=1e-3, 
        nb_epochs=10,
        freq_val=1,
        loss_elt=False,
        save_signals=False,
        save_outputs=False,
        save_model=False,
        save_hist=True
        ):
        """
        Parameters
        ----------
            model_id                   (str): the unique Id of the model.
            experimentation  (Physics class): contains the parameters of the Abel integral
            constraint                 (str): 'cube' of 'slab' to determine the proximal operator
            nb_blocks                  (int): number of blocks in the neural network
            noise                    (float): standard deviation of the Gaussian white noise
            im_set                     (str): images used to construct the 1D signal dataset (Set1 or Set2).
            batch_size                 (int): training batch size.
            train_size                 (int): size of the training dataset.
            val_size                   (int): size of the validation dataset.
            lr_i                     (float): learning rate
            nb_epochs                  (int): number of epochs for training.
            freq_val                   (int): model validation rate.
            dataset_folder             (str): path to the Dataset folder.
            model_folder               (str): path to the pretrained model weigths.
            opt_hist_folder            (str): optimisation history saving folder.
            loss_elt                  (bool): compute the loss in the finite elements space.
            save_signals              (bool): save the 1D signals.
            save_outputs              (bool): save the forwar-backwards parameters.
            save_model                (bool): save the models weights.
            save_hist                 (bool): save the models optimisation history.
        """
        super(FBRestNet, self).__init__() 
        self.model_id = model_id  
        # physical information
        self.physics    = experimentation
        self.noise      = noise
        self.constr     = constraint
        # training information
        self.lr_i       = lr
        self.nb_epochs  = nb_epochs
        self.freq_val   = freq_val
        self.nb_blocks  = nb_blocks
        self.train_size = train_size 
        self.val_size   = val_size
        self.batch_size = batch_size 
        self.im_set     = im_set
        self.loss_fn    = torch.nn.MSELoss(reduction='mean').cuda() if torch.cuda.is_available() else torch.nn.MSELoss(reduction='mean')
        # saving info
        self.model_folder = model_folder
        self.dataset_folder = dataset_folder
        self.opt_hist_folder = opt_hist_folder
        self.save_signals = save_signals
        self.save_outputs = save_outputs
        self.save_model = save_model
        self.save_hist = save_hist
        self.loss_elt   = loss_elt
        # requires regularisation
        self.regul      = (noise>0)&(self.physics.m>20)
        # model creation
        self.model      = MyModel(self.physics,noisy=self.regul,nL=self.nb_blocks,constr=self.constr)
#========================================================================================================
#========================================================================================================
    def LoadParam(self):
        """
        Load the parameters of a trained model (in Trainings)
        """
        path_model = self.model_folder+'param_{}_{}_'.format(
            self.physics.a,self.physics.p
        )+self.constr+'.pt'
        self.model.load_state_dict(torch.load(path_model))
        self.model.eval() # be sure to run this step!
#========================================================================================================
#========================================================================================================    
    def CreateDataSet(self):
        """
        Creates the dataset from an image basis, rescale, compute transformation and noise.
        Construct the appropriate loader for the training and validation sets.
        Parameters
        ----------
            save       (str): 'yes' if the data are saved for reloading.
        Returns
        -------
            (DataLoder): training set
            (DataLoader): validation set
        """
        #
        # Test_cuda()
        # self.device   =
        # self.dtype    =
        # Recuperation des donnees
        nx             = self.physics.nx
        m              = self.physics.m
        a              = self.physics.a
        noise          = self.noise
        nsample        = self.train_size + self.val_size
        im_set         = self.im_set
        Teig           = np.diag(self.physics.eigm**(-a))
        Pelt           = self.physics.Operators()[3]
        # Initialisation
        color          = ('b','g','r')
        #
        liste_l_trsf   = []
        liste_tT_trsf  = []
        #
        save_lisse     = []
        save_l_trsf    = []
        save_blurred   = []
        save_blurred_n = []
        save_tT_trsf   = []
        # Upload Data
        # path : './MyResNet/Datasets/Images/'
        for folder, subfolders, filenames in os.walk(self.dataset_folder+'Images/'+im_set+'/'): 
            for img in filenames: 
                item       = folder+img
                img_cv     = cv.imread(item,cv.IMREAD_COLOR)
                for i,col in enumerate(color):
                    # Etape 1 : obtenir l'histogramme lisse des couleurs images
                    histr  = cv.calcHist([img_cv],[i],None,[256],[0,256]).squeeze()
                    # Savitzky-Golay
                    y      = savgol_filter(histr, 21, 5)
                    # interpolation pour nx points
                    x      = np.linspace(0,1,256, endpoint=True)
                    xp     = np.linspace(0,1,nx,endpoint=True)
                    f      = interp1d(x,y)
                    yp     = f(xp)
                    # normalisation
                    ncrop         = nx//20
                    yp[:ncrop]    = yp[ncrop-1]
                    yp[nx-ncrop:] = 0
                    yp[yp<0]      = 0
                    yp            = yp/np.amax(yp)
                    # filtering high frequencies
                    fmax          = 4*m//5
                    filtre        = Physics(nx,fmax)
                    yp            = filtre.BasisChange(yp)
                    x_true        = filtre.BasisChangeInv(yp)
                    x_true[x_true<0] = 0
                    if self.constr == 'cube':
                        x_true += 0.01
                        x_true  = 0.9*x_true/np.amax(x_true)
                    if self.constr == 'slab':
                        u      = 1/nx**2*np.linspace(1,nx,nx)
                        x_true = 0.5*x_true/np.dot(u,x_true)
                    # reshaping in channelxm
                    x_true  = x_true.reshape(1,-1)
                    # save
                    save_lisse.append(x_true.squeeze())
                    # Etape 2 : passage dans la base de T^*T
                    yp          = self.physics.BasisChange(x_true)
                    x_true_trsf = yp.reshape(1,-1)
                    # save
                    liste_l_trsf.append(x_true_trsf)
                    save_l_trsf.append( x_true_trsf.squeeze())
                    #  Etape 3 : obtenir les images bruitees par l' operateur d' ordre a
                    # transform
                    x_blurred  = self.physics.Compute(x_true).squeeze()
                    # save
                    save_blurred.append(x_blurred)
                    # Etape 4 : noise 
                    #vn          = np.zeros(m)
                    #vn_temp     = np.random.randn(m)*self.physics.eigm**(-2*a)
                    #vn[fmax:]   = vn_temp[fmax:]
                    #vn_elt      = self.physics.BasisChangeInv(vn)
                    #vn_elt      = vn_elt#/np.linalg.norm(vn_elt)
                    # White noise (as indicated in the paper)
                    noise = np.random.randn(nx)
                    x_blurred_n = x_blurred + self.noise*np.linalg.norm(x_blurred)/np.sqrt(nx)*noise#vn_elt
                    # save
                    save_blurred_n.append(x_blurred_n)
                    # Etape 5 : bias
                    #x_b       = self.physics.ComputeAdjoint(x_blurred.reshape(1,-1))
                    #x_b      += (Teig.dot(vn)).reshape(1,-1) # noise
                    # Compute Bias as in the paper T*(y_noisy)
                    x_b = self.physics.ComputeAdjoint(x_blurred_n)
                    x_b       = x_b.reshape(1,-1)
                    # and save
                    liste_tT_trsf.append(x_b)
                    save_tT_trsf.append(x_b.squeeze())
        # Export data in .csv
        if self.save_signals:
            seq = 'a{}_'.format(self.physics.a) + self.constr 
            # initial signal, no noise, elt basis
            np.savetxt(self.dataset_folder+'Signals/data_l_'+seq+'.csv',      save_lisse,   delimiter=', ', fmt='%12.8f')
            # initial signal, no noise, eig basis
            np.savetxt(self.dataset_folder+'Signals/data_l_trsf_'+seq+'.csv', save_l_trsf,  delimiter=', ', fmt='%12.8f')
            # blurred signal, no noise, elt basis
            np.savetxt(self.dataset_folder+'Signals/data_b_'+seq+'.csv',    save_blurred, delimiter=', ', fmt='%12.8f')
            # blurred signal, noisy, elt basis
            np.savetxt(self.dataset_folder+'Signals/data_bn_'+seq+'_n{}'.format(self.noise)+'.csv',  save_blurred_n, delimiter=', ', fmt='%12.8f')
            # Transposed blurred signal, noisy, eig basis
            np.savetxt(self.dataset_folder+'Signals/data_tTb_'+seq+'_n{}'.format(self.noise)+'.csv',  save_tT_trsf, delimiter=', ', fmt='%12.8f')
        # Tensor completion
        x_tensor = torch.FloatTensor(np.array(liste_l_trsf)) # signal in cos/eig basis
        y_tensor = torch.FloatTensor(np.array(liste_tT_trsf))# blurred and noisy signal in elt basis
        #
        dataset = TensorDataset(y_tensor[:nsample], x_tensor[:nsample])
        # Split dataset
        train_dataset, val_dataset = random_split(dataset, [self.train_size, self.val_size])
        #
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # # Plot the output with and without noise
        # plt.plot(save_blurred_n[0],label='Noisy')
        # plt.plot(save_blurred[0],label='Noiseless')
        # plt.legend()
        # plt.title('Output signal')
        # plt.show()
        
        return train_loader, val_loader
#========================================================================================================
#========================================================================================================
    def LoadDataSet(self):
        """
        Creates the appropriate loader for the training and validation sets
        when the dataset is already created.
        Returns
        -------
            (DataLoder): training set
            (DataLoader): validation set
        """
        #
        nsample = self.train_size + self.val_size
        #
        seq = 'a{}_'.format(self.physics.a) + self.constr
        dfl     = pd.read_csv(self.dataset_folder+'Signals/data_l_trsf_'+seq+'.csv', sep=',',header=None)
        dfb    = pd.read_csv(self.dataset_folder+'Signals/data_tTb_'+seq+'_n{}'.format(self.noise)+'.csv', sep=',',header=None)
        _,m     = dfl.shape
        _,nx    = dfb.shape
        #
        x_tensor = torch.FloatTensor(dfl.values[:nsample]).view(-1,1,m)
        y_tensor = torch.FloatTensor(dfb.values[:nsample]).view(-1,1,nx)
        #
        dataset = TensorDataset(y_tensor, x_tensor)
        # Split dataset
        train_dataset, val_dataset = random_split(dataset, [self.train_size, self.val_size])
        #
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)
        #
        return train_loader, val_loader
#========================================================================================================
#========================================================================================================    
    def train(self,train_set,val_set,device='cpu',test_lipschitz=True):
        """
        Trains FBRestNet.
        Parameters
        -------
            train_loader (DataLoder): training set
            val_loader  (DataLoader): validation set
            test_lipschitz    (bool): True if the lipschitz constant is computed during training
        """      
        # to store results
        nb_epochs  = self.nb_epochs
        nb_val     = self.nb_epochs//self.freq_val
        loss_train = np.zeros(nb_epochs)
        loss_val   = np.zeros(nb_val)
        loss_init  = np.zeros(nb_val)
        lip_cste   = np.zeros(nb_val)
        hyper_params_list = []
        # defines the optimizer
        lr_i       = self.lr_i
        optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()),lr=self.lr_i)   
        # filtering parameter
        fmax     = 4*self.physics.m//5
        # trains for several epochs
        for epoch in range(0,self.nb_epochs): 
            # sets training mode
            self.model.train()
            # Do NOT restart the optimizer each epoch!!!!
            # # modifies learning rate
            # if epoch>0:
            #     lr_i      = self.lr_i*0.98 
            #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()), lr=lr_i)
            # TRAINING
            # goes through all minibatches
            for i,minibatch in enumerate(train_set):
                [y, x]    = minibatch    # get the minibatch
                x_bias    = Variable(y,requires_grad=False)
                x_true    = Variable(x,requires_grad=False) 
                # definition of the initialisation tensor
                x_init   = torch.zeros(x_bias.size())
                inv      = np.diag(self.physics.eigm**(2*self.physics.a))
                tTTinv   = MyMatmul(inv)
                x_init   = tTTinv(y) # no filtration of high frequences
                x_init   = Variable(x_init,requires_grad=False)
                # load data on the device (GPU if available)
                x_init, x_bias, x_true = x_init.to(device), x_bias.to(device), x_true.to(device)
                # prediction
                x_pred    = self.model(x_init,x_bias) 
                # Computes and prints loss
                if self.loss_elt:
                    # Compute the loss in the finite elements space
                    x_pred = self.model.Layers[0].Pelt(x_pred)
                    x_true = self.model.Layers[0].Pelt(x_true)
                # Compute the loss
                loss               = self.loss_fn(x_pred,x_true)
                norm               = torch.norm(x_true.detach())
                loss_train[epoch] += torch.Tensor.item(loss/norm)
                # 
                # sets the gradients to zero, performs a backward pass, and updates the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # normalisation
            loss_train[epoch] = loss_train[epoch]/i
            #
            # VALIDATION AND STATS
            if epoch%self.freq_val==0:
                # Validation
                with torch.no_grad():
                # tests on validation set
                    self.model.eval()      # evaluation mode
                    for i,minibatch in enumerate(val_set):
                        [y, x] = minibatch            # gets the minibatch
                        x_true  = Variable(x,requires_grad=False)
                        x_bias  = Variable(y,requires_grad=False)
                        # definition of the initialisation tensor
                        x_init   = torch.zeros(x_bias.size())
                        tTTinv   = MyMatmul(inv)
                        x_init   = tTTinv(y) # no filtration of high frequences
                        x_init   = Variable(x_init,requires_grad=False)
                        # load data on the device (GPU if available)
                        x_init, x_bias, x_true = x_init.to(device), x_bias.to(device), x_true.to(device)
                        # prediction
                        x_pred  = self.model(x_init,x_bias).detach()
                        # computes loss on validation set
                        if self.loss_elt:
                            # Compute the loss in the finite elements space
                            x_pred = self.model.Layers[0].Pelt(x_pred)
                            x_true = self.model.Layers[0].Pelt(x_true)
                            x_init = self.model.Layers[0].Pelt(x_init)
                        # Compute the loss
                        norm    = torch.norm(x_true.detach())
                        loss    = self.loss_fn(x_pred, x_true)
                        loss_in = self.loss_fn(x_init, x_true)
                        loss_val[epoch//self.freq_val] += torch.Tensor.item(loss/norm)
                        loss_init[epoch//self.freq_val]+= torch.Tensor.item(loss_in/norm)
                    # normalisation
                    loss_val[epoch//self.freq_val] = loss_val[epoch//self.freq_val]/i
                    loss_init[epoch//self.freq_val] = loss_init[epoch//self.freq_val]/i
                    # print stat
                    print("epoch : ", epoch," ----- ","validation : ",'{:.6}'.format(loss_val[epoch//self.freq_val]))
                    # Test Lipschitz
                    lip_cste[epoch//self.freq_val] = self.model.Lipschitz()
                    # Get hyperparams at the end of epoch
                    mu_vec = [np.squeeze(self.model.Layers[layer_id].mu).item() for layer_id in range(self.nb_blocks)]
                    tau_vec = [np.squeeze(self.model.Layers[layer_id].gamma_reg[1]).item() for layer_id in range(self.nb_blocks)]
                    lambda_vec = [np.squeeze(self.model.Layers[layer_id].gamma_reg[0]).item() for layer_id in range(self.nb_blocks)]
                    hyper_params = np.stack((np.array(mu_vec),np.array(tau_vec),np.array(lambda_vec)))
                    hyper_params_list.append(hyper_params)
        print("    ----- initial error : ",'{:.6}'.format(loss_init[-1]))
            
        #=======================
        # training is finished
        print('--------------------------------------------')
        print('Training is done.')
        print('--------------------------------------------')
        
        # # Plots
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.plot(np.linspace(0,nb_epochs-1,nb_epochs),loss_train,label = 'train')
        # ax1.plot(np.linspace(0,nb_epochs-1,nb_val),loss_val,label = 'val')
        # ax1.legend()
        # ax2.plot(np.linspace(0,nb_val-1,nb_val),lip_cste,'r-')
        # ax2.set_title("Lipschitz constant")
        # plt.show()
        #
        print("Final Lipschitz constant = ",lip_cste[-1])

        if self.save_hist:
            opt_hist = {
                'lipschitz':lip_cste,
                'loss_train':loss_train,
                'loss_val':loss_val,
                'fb_params':np.array(hyper_params_list)
            }
            np.save(self.opt_hist_folder+self.model_id+'opt_hist.npy', opt_hist)
        
        # Save model
        if self.save_model:
            torch.save(self.model.state_dict(), self.model_folder+self.model_id+'weights.pt')
#========================================================================================================
#========================================================================================================    
    def test(self,data_set):    
        """
        Computes the averaged error of the output on a dataset.
        Parameters
        ----------
            dataset   (Dataloader): the test set
        Returns
        ----------
            (float): averaged error
        """
        # initial
        torch_zeros = Variable(torch.zeros(1,1,self.physics.m),requires_grad=False)
        counter     = 0
        avrg        = 0
        avrg_in     = 0
        # filtering parameter
        fmax     = 4*self.physics.m//5
        # gies through the minibatch
        with torch.no_grad():
            self.model.eval()
            for i,minibatch in enumerate(data_set):
                [y, x] = minibatch            # gets the minibatch
                x_true = Variable(x,requires_grad=False)
                x_bias = Variable(y,requires_grad=False)
                # definition of the initialisation tensor
                x_init   = torch.zeros(x_bias.size())
                inv      = np.diag(self.physics.eigm**(2*self.physics.a))
                tTTinv   = MyMatmul(inv)
                x_init   = tTTinv(y) # no filtration of high frequences
                x_init   = Variable(x_init,requires_grad=False)
                # prediction
                x_pred    = self.model(x_init,x_bias)
                # compute loss
                loss   = torch.Tensor.item(self.loss_fn(x_pred, x_true))
                norm   = torch.Tensor.item(self.loss_fn(torch_zeros, x_true))
                loss_in= torch.Tensor.item(self.loss_fn(x_init, x_true))
                # l_loss.append(loss/norm)
                avrg    += loss/norm
                avrg_in += loss_in/norm
                counter += 1
        # Plots
        xtc = x_true.numpy()[0,0]
        xpc = x_pred.numpy()[0,0]
        xic = x_init.numpy()[0,0]
        xt  = self.physics.BasisChangeInv(xtc)
        xp  = self.physics.BasisChangeInv(xpc)
        xi  = self.physics.BasisChangeInv(xic)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
        fig.suptitle("Prediction results")
        #ax1.plot(xtc,'+',label = 'true')
        #ax1.plot(xpc,'kx',label = 'pred')
        ax1.plot(xtc,label = 'true', linewidth=3)
        ax1.plot(xic,label = 'init')
        ax1.plot(xpc,label = 'pred')
        ax1.set_xlabel('k')
        ax1.legend()
        #ax2.plot(np.linspace(0,1,self.physics.nx),xt,'+',label = 'true')
        ax2.plot(np.linspace(0,1,self.physics.nx),xt,label = 'true', linewidth=3)
        ax2.plot(np.linspace(0,1,self.physics.nx),xi,label = 'init')        
        ax2.plot(np.linspace(0,1,self.physics.nx),xp,label = 'pred')
        ax2.set_xlabel('t')
        ax2.legend()
        plt.show()
        #
        print("Erreur de sortie : ",avrg/counter)
        print("Erreur initiale : ",avrg_in/counter)
        loss_elt = self.loss_fn(self.physics.BasisChangeInv(x_pred), self.physics.BasisChangeInv(x_true))
        init_loss_elt = self.loss_fn(self.physics.BasisChangeInv(x_init), self.physics.BasisChangeInv(x_true))
        norm_elt = self.loss_fn(torch.zeros(1,1,self.physics.nx), self.physics.BasisChangeInv(x_true))
        print('Finite element basis error: {}'.format(loss_elt/norm_elt))
        print('Finite element basis init error: {}'.format(init_loss_elt/norm_elt))
        # return 
        return avrg/counter
#========================================================================================================
#======================================================================================================== 
    def test_gauss(self, noise = 0.05):
        """
        Apply and test the inversion method on a gaussian function.
        Parameters
        ----------
            noise   (float): standard deviation of Gaussian white noise
        
        """
        # Gaussienne 
        nx    = self.physics.nx
        m     = self.physics.m
        a     = self.physics.a
        t     = np.linspace(0,1,nx)
        gauss = np.exp(-(t-0.5)**2/(0.1)**(2))
        # filtering high frequencies
        fmax          = 4*m//5
        filtre        = Physics(nx,fmax)
        gauss         = filtre.BasisChange(gauss)
        gauss         = filtre.BasisChangeInv(gauss)
        gauss[gauss<0] = 0
        #
        if self.constr == 'cube':
            gauss = 0.9*(gauss+0.01)/np.amax(gauss)
        if self.constr == 'slab':
            u      = 1/nx**2*np.linspace(1,nx,nx)
            gauss = 0.5*gauss/np.dot(u,gauss)
        # export
        if self.save_signals:
            Export_Data(t,gauss,self.dataset_folder+'data','gauss_'+self.constr)
        # obtenir les images bruitees par l' operateur d' ordre a
        # transform
        x_blurred  = self.physics.Compute(gauss).squeeze()
        yp         = self.physics.BasisChange(x_blurred)
        # Etape 4 : noise 
        vn          = np.zeros(m)
        vn_temp     = np.random.randn(m)*self.physics.eigm**(-2*a)
        vn[fmax:]   = vn_temp[fmax:]
        vn_elt      = self.physics.BasisChangeInv(vn)
        vn_elt      = vn_elt/np.linalg.norm(vn_elt)
        #vn          = noise*np.linalg.norm(yp)*vn/np.linalg.norm(vn)
        # White noise (as indicated in the paper)
        noise = np.random.randn(nx)
        x_blurred_n = x_blurred + self.noise*np.linalg.norm(x_blurred)/np.sqrt(nx)*noise
        
        # Etape 5 : bias
        #x_b  = self.physics.ComputeAdjoint(x_blurred_n)
        #Teig      = np.diag(self.physics.eigm**(-a))
        #x_b       = self.physics.ComputeAdjoint(x_blurred.reshape(1,-1))
        #x_b      += (Teig.dot(vn)).reshape(1,-1) # noise
        x_b = self.physics.ComputeAdjoint(x_blurred_n)
        x_b       = x_b.reshape(1,-1)

        # passage float tensor
        x_bias    = Variable(torch.FloatTensor(x_b.reshape(1,1,-1)),requires_grad=False)
        # definition of the initialisation tensor
        with torch.no_grad():
        # tests on validation set
            self.model.eval()
            x_init   = torch.zeros(x_bias.shape)
            tTTinv   = MyMatmul(np.diag(self.physics.eigm**(2*self.physics.a)))
            x_init   = tTTinv(x_bias) # filtration of high frequences
            x_init   = Variable(x_init.reshape(1,1,-1),requires_grad=False)
            # prediction
            x_pred   = self.model(x_init,x_bias)
            xpc      = x_pred.detach().numpy()[0,0,:]
            xp       = self.physics.BasisChangeInv(xpc)
            #xp[xp<0] = 0
        # export
        print(type(self.constr))
        if self.save_signals:
            Export_Data(t,xp,self.dataset_folder+'data','gauss_pred_a{}'.format(self.physics.a)+self.constr)
        # plot
        plt.plot(t,gauss,linewidth=3,label='Gaussian')
        plt.plot(t,self.physics.BasisChangeInv(x_init.numpy()[0,0]), label='Init')
        plt.plot(t,xp, label='Pred')
        plt.xlabel('t')
        plt.legend()
        plt.show()

        print("|x-xp|/|x| = ",(np.linalg.norm(xp-gauss)/np.linalg.norm(gauss))**2)


# Import Packages
import numpy as np
#from torch.autograd import Variable
#from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
#import matplotlib as mpl
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import time

# Import model
from FBRN.myfunc import Physics
#from FBRN.myfunc import MyMatmul
from FBRN.main import FBRestNet
#from FBRN.model import MyModel
#from FBRN.myfunc import Export_hyper

def plot_loss(loss_train, loss_val, figs_path_base):
    n_epochs_train = len(loss_train)
    n_val   = len(loss_train)
    # Plots
    plt.figure(figsize=(12,4))
    plt.plot(np.linspace(0,n_epochs_train-1,n_epochs_train),loss_train,label = 'train')
    plt.plot(np.linspace(0,n_epochs_train-1,n_val),loss_val,label = 'val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(figs_path_base+'training_loss.pdf')

def plot_lip(lip_cte, figs_path_base):

    fig = plt.figure(figsize=(10,4))

    ax = fig.add_subplot(2,2,1)
    ax.plot(lip_cte)
    ax.set_ylabel('Linear scale')
    ax.set_title('Lipschitz constant')

    ax = fig.add_subplot(2,2,3)
    ax.plot(lip_cte)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Log scale')
    ax.set_yscale('log')

    ax = fig.add_subplot(2,2,(2,4))
    ax.plot(lip_cte[:15])
    ax.set_title('Zoom on the first epochs')

    plt.savefig(figs_path_base+'lipschitz_constant.pdf')

def plot_fb_params(fb_params, figs_path_base):

    nb_blocks = fb_params.shape[2]
    nb_epochs = fb_params.shape[0]
    mu_vec = fb_params[:,0]
    tau_vec = fb_params[:,1]
    lambda_vec = fb_params[:,2]

    # Plot the final params
    im, ax = plt.subplots(1,3, figsize=(16,4))

    ax[0].plot(mu_vec[-1])
    ax[0].set_xlabel('Layer')
    ax[0].set_title(r'Barrier $\mu$')

    ax[1].plot(tau_vec[-1])
    ax[1].set_xlabel('Layer')
    ax[1].set_title(r'Regularization $\tau$')

    ax[2].plot(lambda_vec[-1])
    ax[2].set_xlabel('Layer')
    ax[2].set_title(r'Stepsize $\lambda$')
    plt.savefig(figs_path_base+'fb_params_final.pdf')

    # Plot the fb params evolution
    im, ax = plt.subplots(1,3, figsize=(16,4))
    for i in range(fb_params.shape[0]):

        ax[0].plot(np.arange(1, nb_blocks+1),mu_vec[i], 
            color=sns.color_palette('inferno',n_colors=nb_epochs)[nb_epochs-i-1],
            label='First epoch' if i==0 else 'Last epoch' if i==(nb_epochs-1) else '_hide'
        )
        ax[0].set_yscale('log')
        ax[0].set_xlabel('Layer')
        ax[0].set_title(r'Barrier $\mu$ (log scale)')

        ax[1].plot(np.arange(1, nb_blocks+1),tau_vec[i], 
            color=sns.color_palette('inferno',n_colors=nb_epochs)[nb_epochs-i-1],
            label='First epoch' if i==0 else 'Last epoch' if i==(nb_epochs-1) else '_hide'
        )
        ax[1].set_xlabel('Layer')
        ax[1].set_title(r'Regularization $\tau$')

        ax[2].plot(np.arange(1, nb_blocks+1),lambda_vec[i], 
            color=sns.color_palette('inferno',n_colors=nb_epochs)[nb_epochs-i-1],
            label='First epoch' if i==0 else 'Last epoch' if i==(nb_epochs-1) else '_hide'
        )
        ax[2].set_xlabel('Layer')
        ax[2].set_title(r'Stepsize $\lambda$')

    for i in range(3): ax[i].legend()
    plt.savefig(figs_path_base+'fb_params_evolution.pdf')

def train_eval_plot(**args):
    
    # Paths
    output_path = args['output_folder']+args['model_id']+'out/'

    # Create output file structure
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(output_path+'figures')
        os.mkdir(output_path+'logs')
        os.mkdir(output_path+'model')
        os.mkdir(output_path+'opt_hist')
        os.mkdir(output_path+'results')

    # Open log file
    old_stdout = sys.stdout
    log_file = open(output_path+'logs/train_eval_plot.log', 'w', 1)
    sys.stdout = log_file

    # Print click options
    print('Click options:')
    for key in args:
        print('\t{0:<18}->'.format(key), args[key])

    # Print GPU info (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device: {}'.format(device))
    if torch.cuda.is_available():
        print('GPU found -> {}'.format(torch.cuda.get_device_name(0)))
    else:
        print('Disclaimer : all GPUs appearing in this work are fictitious. Any resemblance to real GPUs is purely coincidental.')

    # Create the model
    print('Creating the model...')
    model = FBRestNet(
        model_id=args['model_id'],
        dataset_folder=args['dataset_folder'],
        model_folder=output_path+'model/',
        opt_hist_folder=output_path+'opt_hist/',
        experimentation=Physics(
            args['n'],
            args['m'],
            args['a'],
            args['p']
        ),
        nb_blocks=args['nb_blocks'],
        noise=args['noise'],
        constraint=args['constraint'],
        im_set=args['im_set'],
        train_size=args['train_size'],
        val_size=args['val_size'],
        batch_size=args['batch_size'],
        lr=args['lr'], 
        nb_epochs=args['nb_epochs'],
        freq_val=args['freq_val'],
        loss_elt=args['loss_elt'],
        save_signals=args['save_signals'],
        save_outputs=args['save_outputs'],
        save_model=args['save_model'],
        save_hist=args['save_hist']
    )

    # Place the model on the device
    model.to(device)
    model.model.to(device)

    # Create datasets
    print('Generating datsets...')
    train_set, val_set = model.CreateDataSet()

    # Train the model
    if args['train_opt']:
        print('Training the model...')
        start_time = time.time()
        model.train(train_set,val_set,device)
        train_time = time.time() - start_time
        print('Total training time: {}h {}min'.format(int(train_time/3600),int(train_time%3600 /60)))

    # Evaluate the results

    # Generate the plots
    if args['plot_opt']:
        print('Generating figures...')
        figs_path_base = output_path + 'figures/' + args['model_id']
        opt_hist = np.load(
            args['output_folder']+args['model_id']+'out/'+'opt_hist/'+
            args['model_id']+'opt_hist.npy', allow_pickle=True
        )[()]
        # Generate loss plot
        plot_loss(opt_hist['loss_train'], opt_hist['loss_val'], figs_path_base)
        # Generate lipschitz constant evolution plot
        plot_lip(opt_hist['lipschitz'], figs_path_base)
        # Generate forward-backwards params plots
        plot_fb_params(opt_hist['fb_params'], figs_path_base)

    
    # Close the log file
    print('\nDone!')
    sys.stdout = old_stdout
    log_file.close()

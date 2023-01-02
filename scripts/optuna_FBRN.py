# Import packages
import optuna
import logging
from optuna.trial import TrialState
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Import model
from FBRN.myfunc import Physics
from FBRN.main import FBRestNet

# Optimize model hyperparameters:
# lr
# nb_epochs
# nb_blocks

def define_model(hyperparams, **args):

    # Prepare output file structure
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

    args['lr'] = hyperparams['lr']
    args['nb_blocks'] = hyperparams['nb_blocks']
    args['nb_epochs'] = hyperparams['nb_epochs']

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

    model = FBRestNet(
        model_id=args['model_id'],
        dataset_folder=args['dataset_folder'],
        model_folder=output_path+'model/',
        opt_hist_folder=output_path+'opt_hist/',
        results_folder=output_path+'results/',
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

    return model, log_file, old_stdout

def objective(trial, **args):
    
    hyperparams = {
        'lr' : trial.suggest_float('learning_rate', 1e-3, 1e1,log=True),
        'nb_blocks' : trial.suggest_int('nb_blocks', 15, 35),
        'nb_epochs' : trial.suggest_int('nb_epochs', 15,100)
    }

    model, log_file, old_stdout = define_model(hyperparams, **args)
    # Place the model on the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.model.to(device)


    # Create datasets
    print('Generating datsets...')
    train_set, val_set = model.CreateDataSet()
    # Train and evaluate the model
    model.train(train_set,val_set,device)
    loss, _ = model.test(val_set)

    # Close the log file
    print('\nDone!')
    sys.stdout = old_stdout
    log_file.close()

    return loss


def optimize_hyperparams(**args):

    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler(args['output_folder']+args['model_id']+'out/logs/optuna.log', mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

    func = lambda trial:objective(trial, **args)

    study = optuna.create_study(direction='minimize')
    study.optimize(func, n_trials=30)


    
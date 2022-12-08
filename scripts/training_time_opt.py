# Import Packages
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys
import time
import multiprocessing as mp

# Import model
from FBRN.myfunc import Physics
from FBRN.main import FBRestNet

def train_eval_plot(**args):
    
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
    model_id = 'time_test_',
    dataset_folder = '../Datasets/',
    model_folder = '../outputs/models/',
    opt_hist_folder = '../outputs/opt_hist/',
    experimentation=Physics(2000,50,1,1),
    nb_blocks=20,
    im_set="Set1",
    noise = 0.05,        
    constraint = 'cube',  
    train_size=400,
    val_size=200,
    batch_size=1,
    lr=1e-3, 
    nb_epochs=10,
    freq_val=1,
    loss_elt=False,
    save_signals=False,
    save_outputs=False,
    save_model=False,
    save_hist=True
    )

    # Place the model on the device
    model.to(device)
    model.model.to(device)

    # Create datasets
    print('Generating datsets...')
    train_set, val_set = model.CreateDataSet()

    train_set.pin_memory = True
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_set.num_workers = num_workers
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_set, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
    
    print('\nDone!')

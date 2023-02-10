# FBResNet

Forward Backward ResNet. 

This work is a continuation of the work done by Cecile Della Valle in https://github.com/ceciledellavalle/FBResNet.

### Installation
1. Install requirements:

   ```
   $ pip install -r requirements.txt
   ```
2. Install the FBRN model from the FBRN directory:
   ```
   $ pip install .
   ```
3. The model can be tested by using the example notebook in the notebooks folder or by submitting the jobs bash files:
   ```
   $ bash ./jobs/test.sh
   ```

### Files organization
* **`Datasets`**
   * `Images`: two sets of images from which are extracted 1D signal datasets
   * `Signals`: dataset constructed from images
* **`FBRN`**: contains FBRestNet files
   * `info.py`: model information. Allows pip installing the FBRN model.
   * `main.py`: contains FBResNet class, which is the main class including train and test functions
   * `model.py`: includes the definition of the layers in FBRestNet
   * `myfunc.py`: useful functions used such as convolution and Abel operators
	* `proxop`: contains he cardan class used to compute the proximity operator of the logarithmic barrier
	   * `hyperslab.py`: proximity operator for hyperslab constraint
		* `hyperscube.py`: proximity operator for cubic constraint
* **`scripts`**
   * `FBRN_click.py`: interface model parameters and options through a [click](https://click.palletsprojects.com/en/8.1.x/) comand line interface. 
   * `fouriermethod.py`: 1D reconstruction with Fourier filtering method.
   * `lowessmethod.py`: 1D reconstruction with Lowess filtering method.
   * `optuna_FBRN.py`: explore and find the best hyperparameters for the FBRN model. 
   * `train_utils.py`: train, test and plot functions.
   * `training_time_opt.py`: study FBRN trainig times.
* **`notebooks`**: collection of jupyter notebooks for different analysis and studies.
* **`jobs`**: bahs files for submitting model training and evaluation with different parameters.
* **outputs**: contains the predictions and the FBRN weights for different model parameters (*Not in Github*). You can find the output folder as well as the datasets [here!](https://zenodo.org/record/7628457#.Y-YrxS_MK5c)
* **Figures**


import FBRN
from train_utils import train_eval_plot
from optuna_FBRN import optimize_hyperparams
import click

@click.command()
## Script options
@click.option(
    '--train_opt',
    default=True,
    type=bool,
    help='Train the model.'
)
@click.option(
    '--test_opt',
    default=True,
    type=bool,
    help='Test the model.'
)
@click.option(
    '--plot_opt',
    default=True,
    type=bool,
    help='Plot the model results.'
)
## Paths
@click.option(
    '--dataset_folder',
    default='../Datasets/',
    type=str,
    help='Path to the Dataset folder wich contains the images and the saved datasets.'
)
@click.option(
    '--output_folder',
    default='../outputs/',
    type=str,
    help='Path to the output folder.'
)
@click.option(
    '--model_folder',
    default='../outputs/models/',
    type=str,
    help='Path to the pretrained model folder.'
)
@click.option(
    '--opt_hist_folder',
    default='../outputs/opt_hist/',
    type=str,
    help='Path to the optimisation history saving folder.'
)
@click.option(
    '--results_folder',
    default='../outputs/results/',
    type=str,
    help='Path to the test predictions saving folder.'

)
@click.option(
    '--figure_folder',
    default='../outputs/figures/',
    type=str,
    help='Path to the figures folder.'
)
## Physics parameters
@click.option(
    '--a',
    default=1,
    type=float,
    help='Order of ill-posedness.'
)
@click.option(
    '--p',
    default=1,
    type=float,
    help='Order of apriori smothness.'
)
@click.option(
    '--n',
    default=2000,
    type=int,
    help='The number of data points of the signals.'
)
@click.option(
    '--m',
    default=50,
    type=int,
    help='Eigenvector space span size'
)
## Model parameters
@click.option(
    '--model_id',
    default='model_000_',
    type=str,
    help='Model unique identifier.'
)
@click.option(
    '--nb_blocks',
    default=20,
    type=int,
    help='Number of blocks of the FBRestNet.'
)
## Dataset parameters
@click.option(
    '--im_set',
    default='Set1',
    type=str,
    help='Set1 or Set2 to select the set of images to construct the 1D dataset.'
)
@click.option(
    '--noise',
    default=0.05,
    type=float,
    help='Relative noise level of the input signals'
)
@click.option(
    '--constraint',
    default='cube',
    type=str,
    help='Type of constraint for the recovered signal. It can be cube or slab'
)
## Training options
@click.option(
    '--train_size',
    default=50,
    type=int,
    help='Size of the training dataset.'
)
@click.option(
    '--val_size',
    default=10,
    type=int,
    help='Size of the validation dataset.'
)
@click.option(
    '--batch_size',
    default=1,
    type=int,
    help='Training batch size.'
)
@click.option(
    '--lr',
    default=1e-3,
    type=float,
    help='Learning rate.'
)
@click.option(
    '--nb_epochs',
    default=10,
    type=int,
    help='Number of training epochs.'
)
@click.option(
    '--freq_val',
    default=1,
    type=int,
    help='Model validation rate. Set to 1 to validate every epoch.'
)
@click.option(
    '--loss_elt',
    default=False,
    type=bool,
    help='Evaluate the loss in the finite elements space.'
)
## Output options
@click.option(
    '--save_signals',
    default=False,
    type=bool,
    help='Save the 1D signals into csv files.'
)
@click.option(
    '--save_outputs',
    default=False,
    type=bool,
    help='Save the forward backwards parameters.'
)
@click.option(
    '--save_model',
    default=False,
    type=bool,
    help='Save the models weights.'
)
@click.option(
    '--save_hist',
    default=True,
    type=bool,
    help='Save the models optimisation history.'
)
# Optuna option
@click.option(
    '--optuna',
    default=False,
    type=bool,
    help='Runing optuna framework for hyperparameter tuning.'
)
# Optuna parameter range
@click.option(
    '--lr_min',
    default=1e-4,
    type=float,
    help='Optuna minimum learning rate.'
)
@click.option(
    '--lr_max',
    default=False,
    type=float,
    help='Optuna maximum learning rate.'
)
@click.option(
    '--block_min',
    default=20,
    type=int,
    help='Optuna minimum number of blocks.'
)
@click.option(
    '--block_max',
    default=20,
    type=int,
    help='Optuna maximum number of blocks.'
)
@click.option(
    '--epoch_min',
    default=20,
    type=int,
    help='Optuna minimum number of epochs.'
)
@click.option(
    '--epoch_max',
    default=150,
    type=int,
    help='Optuna maximum number of epochs.'
)






def main(**args):
    # check if dataset size is Ok (<600)
    # check physics values
    # other..
    if args['optuna']:
        optimize_hyperparams(**args)
    else:
        train_eval_plot(**args)

if __name__ == "__main__":
    main()
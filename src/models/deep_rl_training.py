"""
Usage:
    1. cd src
    2. python3 models/deep_rl_training.py 
"""

import os, sys
from pathlib import Path
import warnings

# Set environment variable
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:urllib3'
# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

from src.utils import *
from src.data.data_loader import *
from src.models.deep_rl_agent import train_network
from src.models.deep_rl_agent import network_inference
from src.visualization.strategy_evaluation import evaluation

def rl_agent(config_file_simulation,config_file_agent):
    
    """Function that trains the RL agent based on the configuration of the config files
    
    Parameters
    ----------
    config_file_simulation : simulation settings for the JIVR model and the underlying asset 
    config_file_agent : hyperparameters of the RL agent

    Output
    ----------
    deltas: hedging strategies
    
    """
    # 0) Default parameters 
    model = config_file_simulation['model']
    maturity = config_file_simulation["number_days"]   # Maturities: {21,63,126} 
    isput = config_file_agent["isput"]                 # Put or Call: {True,False}
    moneyness = config_file_agent["moneyness"]         # Moneyness options: {"ATM","ITM","OTM"}
    prepro_stock = config_file_agent['prepro_stock']   # Price preprocessing {Log, Log-moneyness, Nothing}
    r = config_file_simulation['r']                    # Annualized continuous risk-free rate
    q = config_file_simulation['q']                    # Annualized continuous dividend yield
    h =  eval(config_file_simulation['size_step'])     # Step size

    # 1) Loading data in the right shape for RL-agent input
    option, n_timesteps, paths, paths_valid, disc_batch, dividend_batch, V_0, V_test, strike = training_variables(model, maturity, moneyness, isput, prepro_stock, r, q)

    # 2) First layer of RL agent hyperparameters 
    network         = "FFNN"
    state_space     = config_file_agent['state_space']      # State space considered in the RL framework {"Full","Reduced_1","Reduced_2"}
    cash_constraint = config_file_agent['cash_constraint']  # Boolean variable to include cash constraints {True,False}
    constraint_max  = config_file_agent['constraint_max']   # Cash constraint limit (positive number)
    nbs_point_traj  = paths.shape[0]                        # time steps 
    batch_size      = config_file_agent['batch_size']       # batch size {296,1000} 
    nbs_input       = paths.shape[2]                        # number of features
    nbs_units       = config_file_agent['nbs_units']        # neurons per layer/cell
    nbs_assets      = 1                                     # number of hedging intruments
    loss_type       = config_file_agent['loss_type']        # loss function {"CVaR","MSE","SMSE"}
    lr              = config_file_agent['lr']               # learning rate of the Adam optimizer
    dropout_par     = config_file_agent['dropout_par']      # dropout regularization parameter 

    # 3) Second layer of RL agent hyperparameters
    transaction_cost = config_file_agent['transaction_cost']  # Proportional transaction cost [0,5/10000,5/1000,1/100]
    riskaversion     = config_file_agent['riskaversion']      # CVaR confidence level (0,1)
    epochs           = config_file_agent['epochs']            # Number of epochs, training iterations 

    # 4) Third layer of parameters
    display_plot    = config_file_agent['display_plot']       # Display plot of training and validation loss
    display_metrics = config_file_agent['display_metrics']    # Display metrics with test set

    # 5) Train RL agent
    loss_train_epoch = train_network(model, network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, paths, V_0, V_test, strike, disc_batch,
        dividend_batch, transaction_cost, riskaversion, paths_valid, epochs, display_plot, option, moneyness, isput, prepro_stock)

    # 6) Compute matrics based on test set
    # 6.1) Compute hedging strategy
    deltas = network_inference(model, network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, V_test, strike, disc_batch,
        dividend_batch, transaction_cost, riskaversion, paths_valid, option, moneyness, isput, prepro_stock, False)
    print("--- Deep agent trained and stored in ../models/.. ---")
    # 6.2) Assess hedging strategy with test set
    if display_metrics == True:
        results = evaluation(model,deltas,riskaversion,r,q,h,strike)
        #df_cost_functions.to_csv("Results.csv")
        print("--- Hedging startegy stored in ../results/Trining/.. ---")
        # Transpose the DataFrame
        results = results.transpose()
        # Rename the columns
        results.columns = ['Metrics']
        results['Metrics'] = results['Metrics'].apply(lambda x: round(x, 6) if isinstance(x, (int, float)) else x)
        print("---------------------- Results -----------------------__")
        print(results)


    return deltas
    
if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_agent.yml'))
    config_file_agent = config_file["agent"]
    config_file = load_config(os.path.join(main_folder,'cfgs','config_simulation.yml'))
    config_file_simulation = config_file["simulation"]
    _ = rl_agent(config_file_simulation,config_file_agent)
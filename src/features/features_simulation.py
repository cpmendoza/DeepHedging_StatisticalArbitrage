"""
Usage:
    1. cd src
    2. python3 features/features_simulation.py 
"""

import os, sys
from pathlib import Path
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import pandas as pd
import numpy as np
import time
import random
from scipy.stats import norm
from scipy.stats import norminvgauss
from src.utils import *
from random import sample
from scipy.stats import norm
from arch import arch_model
from tqdm import tqdm

class market_simulator(object):
    
    """Class to the implied volatility surface model, recalibrate it, and use it to predict future dynamics
    
    Parameters
    ----------
    config_file : simulation settings for the JIVR model and the underlying asset and 
    
    """
    
    def __init__(self,config_file):
        # Parameters
        
        self.config_file = config_file
        self.model = config_file['simulation']['model']                           #Model to simulate market dynamics
        self.simulations = config_file['simulation']['number_simulations']        #Number of simulated paths
        self.time_steps = config_file['simulation']['number_days']                #Path time-steps
        self.delta = eval(config_file['simulation']['size_step'])                 #Daily time step
        self.q = config_file['simulation']['q']                                   #Dividend rate as a constant
        self.r = config_file['simulation']['r']                                   #Interest rate as a constant
        self.S_0 = config_file['simulation']['stock_price']                       #Initial value of the stock price
        self.K = config_file['simulation']['strike']                              #Strike price
        self.seed = config_file['simulation']['seed']                             #Seed to ensure replicability 
    
    def gjr_garch_simulation(self, mu, omega, alpha_1, beta_1, gamma_1, initial_return, initial_variance, last_residual, num_paths, num_steps):

        """Function that simulates market dynamics based on one of the two econometric models  

        Parameters
        ----------
        [mu, omega, alpha_1, beta_1, gamma_1] : single values - parameters of the econometric model
        initial_return : single value - Last fitted return
        initial_variance : single value - Last fitted conditional variance
        last_residual : single value - Last fitted residual
        num_paths : single value - Number of paths to be simulated
        num_steps : single value - Number of time-steps for the simulation 

        Returns
        -------
        returns : numpy array - simulated paths of the return based on the econometric model
        conditional_variance : numpy array - simulated paths of the conditional variance based on the econometric model

        """

        returns = np.zeros((num_paths, num_steps+1))
        conditional_variance = np.zeros((num_paths, num_steps+1))

        #Initialize
        returns[:, 0] = initial_return
        conditional_variance[:, 0] = initial_variance

        for t in range(1, num_steps+1):
            # Calculate indicator function
            indicator =  (last_residual < 0)*1
            # Calculate conditional variance
            conditional_variance[:, t] = omega + (alpha_1 + gamma_1 * indicator) * (last_residual**2) + beta_1 * conditional_variance[:, t-1]
            # Generate standardized random shocks
            z = np.random.normal(0, 1, num_paths)
            # Noise 
            last_residual = np.sqrt(conditional_variance[:, t]) * z
            # Generate returns
            returns[:, t] = mu + last_residual

        return returns[:,1:], conditional_variance
        
    def simulator(self):
    
        """Function that provide multidimensional vector for updating the parameters 

        Parameters
        ----------
        parameters : data frame - parameters of the econometric model
        betas_series : data frame - historical data of the beta parameters
        h_series : data frame historical data of the volvol of the econometric model
        Y_series : data frame - historical data of the returns

        Returns
        -------
        beta_t : beta parameters at time t
        beta_t_minus : beta parameters at time t-1
        h_t : simulated volatility at time t
        e_t : NIG values at time t

        """

        #Seed to ensure reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

        #Generation of NIG values with Gaussian copula, reference: François et al. (2023).
        directory = os.path.join(main_folder,'data','raw','SP500_random.csv')
        Y_series = pd.read_csv(directory, index_col=0).iloc[5031:,:] # 04/01/16 to 31/12/20

        # Calculate log returns for the 'Close' column
        Y_series['Log Returns'] = np.log(Y_series['Close'] / Y_series['Close'].shift(1))

        # Drop NaN values that result from the shift operation
        Y_series = Y_series.dropna()

        print("-- Market dynamics simulation starts --")

        if self.model == 'Black-Scholes':
            #Simulation of stock prices
            sigma = Y_series['Log Returns'].std()*np.sqrt(252)
            mu = Y_series['Log Returns'].mean()*np.sqrt(252) + (sigma**2)/2
            #print(sigma,mu)
            T  = self.time_steps/252  # Time-to-maturity of the vanilla put option
            h  = T / self.time_steps  # step-size
            Stock_paths = np.zeros((self.time_steps+1, self.simulations))  # matrix of simulated stock prices 
            Stock_paths[0,:] = self.S_0
            for i in range(self.simulations):
                rand_stdnorm     = np.random.randn(self.time_steps)
                Stock_paths[1:,i]  = self.S_0 * np.cumprod(np.exp((mu-sigma**2/2)*h+sigma*np.sqrt(h)*rand_stdnorm))
            Stock_paths = np.transpose(Stock_paths)
            #BS deltas
            time_to_maturity = np.array(sorted((np.arange(self.time_steps)+1),reverse=True))/252
            d1 = (np.log(Stock_paths[:,:-1]/self.K)+((self.r-self.q+(sigma**2)/2)*time_to_maturity))/(sigma*np.sqrt(time_to_maturity))
            deltas = norm.cdf(d1)*np.exp(-1*self.q*time_to_maturity)

            d1 = d1[0,0]
            d2 = d1-sigma*np.sqrt(T)
            option_price = self.S_0*np.exp(-self.q*T)*norm.cdf(d1)-self.K*np.exp(-self.r*T)*norm.cdf(d2)
            
            np.save(os.path.join(main_folder,'data','processed','training',f"Stock_paths__random_f_{self.time_steps}_{self.model}.npy"),Stock_paths)
            np.save(os.path.join(main_folder,'data','processed','training',f"H_simulation__random_f_{self.time_steps}_{self.model}.npy"),sigma) 
            np.save(os.path.join(main_folder,'data','processed','training',f"Deltas__random_f_{self.time_steps}_{self.model}.npy"),deltas[400000:,:])
            np.save(os.path.join(main_folder,'data','processed','training',f"Option_price__random_f_{self.time_steps}_{self.model}.npy"),option_price) 

        else:
            #Define time series
            log_returns = Y_series['Log Returns'].dropna()
            o_ = 0 if self.model == 'GARCH' else 1
            # Fit the GJR-GARCH(1,1) model
            model_garch = arch_model(log_returns, vol='Garch', p=1, o=o_, q=1,rescale=False)
            garch_fit = model_garch.fit(disp='off')
            #print(garch_fit.summary())

            # Parameters of the GJR-GARCH(1,1) model
            mu = garch_fit.params['mu']
            omega = garch_fit.params['omega']
            alpha_1 = garch_fit.params['alpha[1]']
            beta_1 = garch_fit.params['beta[1]']
            gamma_1 = 0 if self.model == 'GARCH' else garch_fit.params['gamma[1]']

            # Initial values
            initial_return = log_returns.iloc[-1]
            initial_variance = garch_fit.conditional_volatility.iloc[-1]**2
            z = np.random.normal(0, 1, self.simulations)
            last_residual = np.sqrt(initial_variance) * z

            # Simulate paths
            returns, conditional_variance = self.gjr_garch_simulation(mu, omega, alpha_1, beta_1, gamma_1, initial_return, initial_variance, last_residual, self.simulations, self.time_steps)

            #Simulate Stock Paths
            Stock_paths = self.S_0*np.cumprod(np.exp(returns),axis=1)
            Stock_paths = np.insert(Stock_paths,0,self.S_0,axis=1)

            #Undelying asset volatility
            volatility = np.sqrt(conditional_variance[:,:])*np.sqrt(252)
            sigma = volatility[:,:-1]

            np.save(os.path.join(main_folder,'data','processed','training',f"Stock_paths__random_f_{self.time_steps}_{self.model}.npy"),Stock_paths)
            np.save(os.path.join(main_folder,'data','processed','training',f"H_simulation__random_f_{self.time_steps}_{self.model}.npy"),volatility) 
            #np.save(os.path.join(main_folder,'data','processed','training',f"Returns_random__random_f_{self.time_steps}.npy"),returns_random[random_sample,:])
            
            #Deltas BS
            #time_to_maturity = np.array(sorted((np.arange(self.time_steps)+1),reverse=True))/252
            #d1 = (np.log(Stock_paths[:,:-1]/self.K)+((self.r-self.q+(sigma**2)/2)*time_to_maturity))/(sigma*np.sqrt(time_to_maturity))
            #deltas = norm.cdf(d1)*np.exp(-1*self.q*time_to_maturity)

            #Deltas GARCH
            garch_neutral = garch_neutral_dynamics(self.r ,mu, omega, alpha_1, beta_1, gamma_1)
            deltas = garch_neutral.array_delta(Stock_paths[400000:,:], volatility[400000:,:], self.K, 1000)
            np.save(os.path.join(main_folder,'data','processed','training',f"Deltas__random_f_{self.time_steps}_{self.model}.npy"),deltas)

            #Option_price GARCH
            Stock_paths = garch_neutral.gjr_garch_simulation_neutral_measure(self.S_0, initial_variance, 100000, self.time_steps)
            option_price = garch_neutral.option_price(Stock_paths)
            np.save(os.path.join(main_folder,'data','processed','training',f"Option_price__random_f_{self.time_steps}_{self.model}.npy"),option_price) 

        print("-- Simulation completed - Features stored in ../data/processed/--")

        return
    
if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    main_folder = str(Path.cwd().parent)
    sys.path.append(main_folder)
    config_file = load_config(os.path.join(main_folder,'cfgs','config_simulation.yml'))
    jivr_model = market_simulator(config_file)
    jivr_model.simulator()
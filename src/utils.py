import logging
import yaml
from tqdm import tqdm
import warnings
import time
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def load_config(config_file):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config

class garch_neutral_dynamics(object):

    """Class to use GARCH neutral dynamics:
        -Paths simulation
        -Option pricing 
        -Delta computation 
    
    Parameters
    ----------
    config_file : simulation settings for the JIVR model and the underlying asset and 
    
    """
    
    def __init__(self,r ,mu, omega, alpha_1, beta_1, gamma_1):

        # Parameters
        self.r = r/252          #Daily risk-free rate
        self.q = 0.01772245/252     #Daily dividend yield
        self.mu = mu            #GARCH model parameter
        self.omega = omega      #GARCH model parameter
        self.alpha_1 = alpha_1  #GARCH model parameter
        self.beta_1 = beta_1    #GARCH model parameter
        self.gamma_1 = gamma_1  #GARCH model parameter

    def gjr_garch_simulation_neutral_measure(self, S, initial_variance, num_paths, num_steps):

        """Function that simulates market dynamics based on one of the two econometric models  

        Parameters
        ----------
        S: initial underlying asset price
        initial_variance : single value - Last fitted conditional variance
        num_paths : single value - Number of paths to be simulated
        num_steps : single value - Number of time-steps for the simulation 

        Returns
        -------
        returns : numpy array - simulated paths of the return based on the econometric model
        conditional_variance : numpy array - simulated paths of the conditional variance based on the econometric model

        """

        conditional_variance = np.zeros((num_paths, num_steps+1))
        returns = np.zeros((num_paths, num_steps))

        #Initialize
        conditional_variance[:, 0] = initial_variance

        #New residual - risk free measure
        nabla_t = (self.mu-self.r+self.q+initial_variance/2)/np.sqrt(initial_variance)
        #generate modify shocks considering the risk netural measure
        z_t = np.random.normal(0, 1, num_paths)-nabla_t
        #residual 
        last_residual = np.sqrt(initial_variance) * z_t

        for t in range(1, num_steps+1):
            # Calculate indicator function
            indicator =  (last_residual < 0)*1
            # Calculate conditional variance
            conditional_variance[:, t] = self.omega + (self.alpha_1 + self.gamma_1 * indicator) * (last_residual**2) + self.beta_1 * conditional_variance[:, t-1]
            # Generate standardized random shocks
            nabla_t = (self.mu-self.r+self.q+conditional_variance[:, t]/2)/np.sqrt(conditional_variance[:, t])
            z_t = np.random.normal(0, 1, num_paths)-nabla_t
            # Noise 
            last_residual = np.sqrt(conditional_variance[:, t]) * z_t
            # Generate returns
            returns[:, t-1] = self.mu + last_residual

        # Simulated paths 
        S_0 = S
        Stock_paths = S_0*np.cumprod(np.exp(returns),axis=1)
        Stock_paths = np.insert(Stock_paths,0,S_0,axis=1)

        return Stock_paths
    
    def option_price(self, S):

        """Function that compute the option price based on Monte-Carlo simulation 

        Parameters
        ----------
        S: numpy array - Simulated paths of the underlying asset

        Returns
        -------
        option_price : single value
        """

        t = S.shape[1]-1
        option_price = (np.maximum(S[:,-1]-100,0)*np.exp(-self.r*t/252)).mean()
        return option_price
    
    def delta_garch(self, S, initial_variance, strike, num_paths, num_steps):

        """Function that compute delta at a given state

        Parameters
        ----------
        S: numpy array - initial_value
        num_paths : single value - Number of paths to be simulated
        num_steps : single value - Number of time-steps for the simulation 

        Returns
        -------
        delta : single value - Position in the underlying asset
        """
        
        Stock_paths = self.gjr_garch_simulation_neutral_measure(S, initial_variance, num_paths, num_steps)
        t = Stock_paths.shape[1]-1
        delta = np.exp(-t*self.r)*((Stock_paths[:,-1]/Stock_paths[:,0])*(Stock_paths[:,-1]>strike)).mean()

        return delta
    
    def vectorial_delta(self, Stock_paths, initial_variance, strike, num_paths, num_steps):

        """Function that compute delta at a given state for a vector

        Parameters
        ----------
        Stock_paths: numpy array - vector of underlying asset prices
        initial_variance: numpy array - vector of underlying asset conditional variances
        strike: Strike price
        num_paths : single value - Number of paths to be simulated
        num_steps : single value - Number of time-steps for the simulation 

        Returns
        -------
        delta : numpy array - vector of estimated deltas
        """

        delta_1 = np.nan
        result = list()
        for element in tqdm(range(Stock_paths.shape[0])):
            attempt = 0
            while attempt < 3:
                try:
                    delta_1 = self.delta_garch(Stock_paths[element],initial_variance[element], strike, num_paths, num_steps)
                    break
                except Exception as e:
                    attempt += 1
            result.append(delta_1) 
        
        return result
    
    def array_delta(self, Stock_paths, conditional_volatility, strike, num_paths_estimation):

        volatility = (conditional_volatility**2)/252
        deltas = np.zeros((Stock_paths.shape[0],Stock_paths.shape[1]-1))
        time_steps = Stock_paths.shape[1]-1
        for time_step in range(time_steps):
            print(f"---- Deltas computation at step {time_step} ----")
            num_steps = time_steps-time_step
            delta_t = self.vectorial_delta(Stock_paths[:,time_step], volatility[:,time_step], strike, num_paths_estimation, num_steps)
            deltas[:,time_step] = delta_t
        
        return deltas
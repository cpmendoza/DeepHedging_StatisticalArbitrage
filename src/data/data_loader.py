"""
Usage:
    1. cd src
    2. python data/data_loader.py
"""

import os, sys
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os, sys
from pathlib import Path
from scipy.stats import norm
import numpy as np

class portfolio_value():
    
    def Black_Scholes_price(self,S, K, TT, sigma, isput,r,q):
        
        """Function that provides the vanilla option price based on Black-Scholes formula

            Parameters
            ----------
            S     : sigle value - Moneyness
            K     : single value - Time to maturity
            TT    : numpy.ndarray - parameters
            sigma : location of a fast convexity change in the IV term structure with respect to time to maturity
            isput : single value - Maximal maturity represented by the model
            r     : annualized risk-free rate
            q     : annualized dividend yield

            Returns
            -------
            price : Option price

        """

        d1 = (np.log(S/K)+(r-q+(sigma**2)/2)*TT)/(sigma*np.sqrt(TT))
        d2 = d1-sigma*np.sqrt(TT)
        if isput == False:
            price = S*np.exp(-q*TT)*norm.cdf(d1)-K*np.exp(-r*TT)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*TT)*norm.cdf(-d2)-S*np.exp(-q*TT)*norm.cdf(-d1)
        return price
    
    def volatility(self,M,tau,beta,T_conv = 0.25,T_max = 5):

        """Function that provides the volatility surface model

            Parameters
            ----------
            M          : sigle value - Moneyness
            tau        : single value - Time to maturity
            beta       : numpy.ndarray - parameters
            T_conv     : location of a fast convexity change in the IV term structure with respect to time to maturity
            T_max      : single value - Maximal maturity represented by the model

            Returns
            -------
            volatility : Implied volatility

        """

        Long_term_ATM_Level = beta[:,0]
        Time_to_maturity_slope = beta[:,1]*np.exp(-1*np.sqrt(tau/T_conv))
        M_plus = (M<0)*M
        Moneyness_slope = (beta[:,2]*M)*(M>=0)+ (beta[:,2]*((np.exp(2*M_plus)-1)/(np.exp(2*M_plus)+1)))*(M<0)
        Smile_attenuation = beta[:,3]*(1-np.exp(-1*(M**2)))*np.log(tau/T_max)
        Smirk = (beta[:,4]*(1-np.exp((3*M_plus)**3))*np.log(tau/T_max))*(M<0)

        volatility = np.maximum(Long_term_ATM_Level + Time_to_maturity_slope + Moneyness_slope + Smile_attenuation + Smirk,0.01)

        return volatility

    def portfolio(self, Price_mat, strike, T, isput, r, q, betas, sigma = None):
        
        """Function that provides the initial price of a portfolio of options

            Parameters
            ----------
            Price_mat : Underlying asset price
            K         : Strike of the option
            TT        : Time-to-maturity
            sigma     : volatility
            isput     : Parameter to determine Put or Call
            r         : annualized risk-free rate
            q         : annualized dividend yield

            Returns
            -------
            price     : Option price of all paths at time zero

        """
        forward_price_1 = Price_mat[0,:]*np.exp(T*(r-q))
        M               = np.log(forward_price_1/strike)*(1/np.sqrt(T))
        if sigma is None:
          sigma         = self.volatility(M,T,betas[:,0,:])
        
        price           = self.Black_Scholes_price(Price_mat[0,:], strike, T, sigma, isput,r,q)
        return price
    

def data_sets_preparation(model, moneyness,isput,prepro_stock, Price_mat, h_simulation, r, q):
    
    """Function that loads the sets to create the training set

        Parameters
        ----------
        moneyness       : Moneyness option {"ATM","ITM","OTM"}
        isput           : Put or Call {True,False}
        prepro_stock    : Price preprocessing {"Log", "Log-moneyness", "Nothing"}
        Price_mat       : Matrix of underlying asset prices
        h_simulation    : Volatility of the underlying asset

        Returns
        -------
        option          : Option name {"Call","Put"}
        n_timesteps     : Time steps
        train_input     : Training set (normalized stock price and features)
        test_input      : Validation set (normalized stock price and features)
        disc_batch      : Risk-free rate update factor exp(h*r)
        dividend_batch  : Dividend yield update factor exp(h*d)
        V_0_train       : Initial portfolio value for training set
        V_0_test        : Initial portfolio value for validation set
        strike          : Strike value of the option hedged 

      """

    # 1) Define environment in terms of the option
    # 1.1) Option type
    moneyness_list = ["ATM","ITM","OTM"]
    K_firsts    = [100,90,110]
    idx         = moneyness_list.index(moneyness)
    if isput == False:
      option = 'Call'
    else:
      option = 'Put'
    # 1.2) Moneyness and strike
    if isput == True:
      if moneyness_list[idx]=="ITM":
        strike   = K_firsts[idx+1]
      elif moneyness_list[idx]=="OTM":
        strike   = K_firsts[idx-1]
      else:
        strike   = K_firsts[idx]
    else:
      strike     = K_firsts[idx]

    n_timesteps  = Price_mat.shape[0]-1      # Daily hedging
    T = n_timesteps/252                      # Time-to-maturity of the vanilla put option (63/252, 252/252)
    h = T / n_timesteps                      # Daily size step (T / n_timesteps) #Modify size of the step

    # Apply a transformation to stock prices
    S = np.transpose(Price_mat)[:,0]
    if(prepro_stock == "Log"):
        Price_mat = np.log(Price_mat)
    elif(prepro_stock == "Log-moneyness"):
        Price_mat = np.log(Price_mat/strike)

    # Construct the train and test sets
    # - The feature vector for now is [S_n, T-t_n]; the portfolio value V_{n} will be added further into the code at each time-step
    
    num_features = 2 if model == 'Black-Scholes' else 3

    train_input     = np.zeros((n_timesteps+1, 400000,num_features))
    test_input      = np.zeros((n_timesteps+1, 100000,num_features))
    time_to_mat     = np.zeros(n_timesteps+1)
    time_to_mat[1:] = T / (n_timesteps)      # [0,h,h,h,..,h]
    time_to_mat     = np.cumsum(time_to_mat) # [0,h,2h,...,Nh]
    time_to_mat     = time_to_mat[::-1]      # [Nh, (N-1)h,...,h,0]

    train_input[:,:,0] = Price_mat[:,0:400000]
    train_input[:,:,1] = np.reshape(np.repeat(time_to_mat, train_input.shape[1], axis=0), (n_timesteps+1, train_input.shape[1]))
    
    test_input[:,:,0]  = Price_mat[:,400000:]
    test_input[:,:,1]  = np.reshape(np.repeat(time_to_mat, test_input.shape[1], axis=0), (n_timesteps+1, test_input.shape[1]))
    
    if model != 'Black-Scholes':
      train_input[:,:,2] = np.transpose(h_simulation)[:,0:400000]
      test_input[:,:,2] = np.transpose(h_simulation)[:,400000:]
            
    disc_batch            = np.exp(r*h)   # exp(rh)
    dividend_batch        = np.exp(q*h)   # exp(qh)

    #Initial portfolio values
    price          = np.load(os.path.join(f"Option_price__random_f_{n_timesteps}_{model}.npy"))
    V_0_train      = np.ones(train_input.shape[1])*price
    V_0_test       = np.ones(test_input.shape[1])*price
    
    return option, n_timesteps, train_input, test_input, disc_batch, dividend_batch, V_0_train, V_0_test, strike

def load_standard_datasets(model, maturity):
    
    """Function that loads the sets to create the training set

        Parameters
        ----------
        maturity     : time to maturity of the options

        Returns
        -------
        Price_mat    : Matrix of underlying asset prices
        S            : Transposed matrix of the underlying asset price
        betas        : Coefficients of the IV surface 
        h_simulation : Volatility of the underlying asset

      """
    
    # 1) Load datasets to train and test deep hedging algorithm (Simulated paths, Betas IV, volatilities)
    # 1.1) matrix of simulated stock prices
    Price_mat    = np.transpose(np.load(os.path.join(f"Stock_paths__random_f_{maturity}_{model}.npy")))
    # 1.3) Volatility
    h_simulation = np.load(os.path.join(f"H_simulation__random_f_{maturity}_{model}.npy"))
  
    return Price_mat, h_simulation

def training_variables(model, maturity, moneyness, isput, prepro_stock, r, q):

    """Function that loads the sets to create the training set

        Parameters
        ----------
        maturity        : time to maturity of the options
        moneyness       : Moneyness option {"ATM","ITM","OTM"}
        isput           : Put or Call {True,False}
        prepro_stock    : Price preprocessing {"Log", "Log-moneyness", "Nothing"}
        backtest        : Parameter for Backtest procedure {True,False}

        Returns
        -------
        option          : Option name {"Call","Put"}
        n_timesteps     : Time steps
        train_input     : Training set (normalized stock price and features)
        test_input      : Validation set (normalized stock price and features)
        disc_batch      : Risk-free rate update factor exp(h*r)
        dividend_batch  : Dividend yield update factor exp(h*d)
        V_0_train       : Initial portfolio value for training set
        V_0_test        : Initial portfolio value for validation set
        strike          : Strike value of the option hedged 

      """
    owd = os.getcwd()
    os.chdir(os.path.join(main_folder, f"data/processed/Training/"))

    Price_mat, h_simulation = load_standard_datasets(model, maturity)
    option, n_timesteps, train_input, test_input, disc_batch, dividend_batch, V_0_train, V_0_test, strike = data_sets_preparation(model, moneyness,isput,prepro_stock, Price_mat, h_simulation, r, q)

    #change dir back to original working directory (owd)
    os.chdir(owd)
      
    return option, n_timesteps, train_input, test_input, disc_batch, dividend_batch, V_0_train, V_0_test, strike
  


        
        





























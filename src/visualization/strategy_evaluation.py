"""
Usage:
    1. cd src
    2. python models/matchmaking/advisor_to_company_match.py
"""


import os, sys
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro
from scipy import stats


class strategy_hedging_valuation(object):
    
    """Class to implement the delta-hedging framework
    
    Parameters
    ----------
    Time_steps : int, optional
    number_sumlations: int, optional
        
    """
    
    def __init__(self, hedging_strategy = "one_instrument", transaction_cost = 0, isput=False, q = 0.01772245, r = 0.026623194):

        # Parameters
        self.r = r                                    #Interest rate as a constant
        self.q = q                                    #Dividend yield
        self.transaction_cost = transaction_cost
        self.hedging_strategy = hedging_strategy
        self.isput = isput
        
    
    def hedging_error_vector(self, K, Stock_paths, option_price, position_underlying_asset):
        
        #General values
        time_steps = Stock_paths.shape[1]-1
        number_simulations = Stock_paths.shape[0]
        
        #Hedging portfolio
        hedging_portfolio = np.zeros([number_simulations,(time_steps+1)])
        cost_matrix = np.zeros([number_simulations,(time_steps)])
        hedging_portfolio[:,0] = option_price
        
        if self.isput==False:
            payoff = np.maximum(Stock_paths[:,time_steps]-K,0)
        else:
            payoff = np.maximum(K-Stock_paths[:,time_steps],0)
        
        #Compute P&L for the hedging startegies
        if self.hedging_strategy == "one_instrument":
            
            #Computing hedging errors with only one hedging instrument
            for time_step in range(time_steps):
                #Transacition cost computation
                if time_step == 0:
                    cost = np.abs(Stock_paths[:,time_step]*position_underlying_asset[:,time_step]*self.transaction_cost)
                    cost_matrix[:,0] = cost
                else:
                    cost = np.abs(self.transaction_cost*Stock_paths[:,time_step]*(position_underlying_asset[:,time_step]-position_underlying_asset[:,time_step-1]))
                    cost_matrix[:,time_step] = cost_matrix[:,time_step-1] + cost*np.exp((-1)*self.q*(time_step/252))

                phi_0 = hedging_portfolio[:,time_step] - Stock_paths[:,time_step]*position_underlying_asset[:,time_step] - cost
                hedging_portfolio[:,(time_step+1)] = phi_0*np.exp(self.r/252) + position_underlying_asset[:,time_step]*Stock_paths[:,(time_step+1)]*np.exp(self.q/252)
            
            #P&L Computation
            hedging_error_limit = hedging_portfolio[:,time_steps] - payoff
            cost_limit = cost_matrix[:,-1]
 
        return hedging_error_limit, cost_limit, hedging_portfolio
    
def statistics_2(hedging_err):

    "Mean"
    loss = np.mean(hedging_err)
    for x in [0.5,0.75,0.9,0.95,0.99]:
        loss = np.append(loss,np.mean(np.sort(hedging_err)[int(x*hedging_err.shape[0]):]))
    #np.append(loss,np.mean(np.sort(hedging_err)[int(0.95*hedging_err.shape[0]):]) + np.sqrt(np.mean(np.square(hedging_err))))

    return(loss)

def regression_approach(deltas,deep_hedging):
    correlation_time = []
    results = []
    for t in range(1,63):
        x_1 = deltas[:,t]
        y_1 = deep_hedging[:,t]#-deltas[:,t]
        correlation, p_value = stats.spearmanr(x_1, y_1)
        correlation_time.append(correlation)

        # Remove NaN values
        cleaned_data = [x for x in correlation_time if not np.isnan(x)]

        # Compute mean
        mean_value = np.mean(cleaned_data)

        # Add intercept to X
        X = sm.add_constant(x_1)

        # Fit the linear regression model
        model = sm.OLS(y_1.reshape(-1,1), X).fit()

        # Extracting confidence intervals
        conf = model.conf_int()

        # Separating the lower and upper limits
        lower_limits = conf[:, 0]
        upper_limits = conf[:, 1]

        coefficients = model.params
        r_squared = model.rsquared
        p_value = model.f_pvalue
        residuals_mean = model.resid.mean()

        # Compute the Durbin-Watson statistic
        dw_statistic = durbin_watson(model.resid)

        # Perform Breusch-Pagan test for heteroscedasticity
        bp_test = het_breuschpagan(model.resid, model.model.exog)

        # bp_test returns a tuple with Lagrange multiplier statistic, p-value, f-value, and f p-value
        bp_test_stat, bp_test_pvalue, _, _ = bp_test

        # Perform Shapiro-Wilk test for normality of residuals
        shapiro_test_stat, shapiro_test_pvalue = shapiro(model.resid)

        result = {
            'Coefficient_0': coefficients[0],
            'Coefficient_1': coefficients[1],
            'Coefficient_0_inf': lower_limits[0],
            'Coefficient_0_sup': upper_limits[0],
            'Coefficient_1_inf': lower_limits[1],
            'Coefficient_1_sup': upper_limits[1],
            'R-squared': r_squared,
            'P-value': p_value,
            'Residuals Mean': residuals_mean,
            'Durbin-Watson': dw_statistic, #Close to 2 no autocorrelation
            'Breusch-Pagan': bp_test_pvalue,
            'Shapiro': shapiro_test_pvalue
        }
        
        # Append result dictionary to results list
        results.append(result)

    # Create a DataFrame from the results list
    df_results = pd.DataFrame(results)

    # Remove NaN values
    cleaned_data = [x for x in correlation_time if not np.isnan(x)]

    # Compute mean
    mean_value = np.mean(cleaned_data)

    # Display the DataFrame
    return  mean_value, df_results

def evaluation(model,deep_hedging,riskaversion,r,q,h,strike,hedging_strategy = "one_instrument", transaction_cost = 0, isput = False):
    
    #Change directory to load data
    owd = os.getcwd()
    os.chdir(os.path.join(main_folder, f"data/processed/Training/"))

    #Define hyperparameters of hedging valuation
    new_evaluation  = strategy_hedging_valuation(hedging_strategy,transaction_cost,isput)

    #Define general parameters
    discount_factor = np.exp(-1*r*h)
    divided_factor = np.exp(q*h)
    df_summary = pd.DataFrame()
    df_summary_risk = pd.DataFrame()
    df_coefficients = pd.DataFrame()
    
    #Loading data
    Stock_paths = np.load(os.path.join(f"Stock_paths__random_f_63_{model}.npy"))[400000:,:]
    deltas = np.load(os.path.join(f"Deltas__random_f_63_{model}.npy"))
    option_price = np.load(os.path.join(f"Option_price__random_f_63_{model}.npy"))

    #Cleaning wrong simulations
    rows = np.where(np.isnan(deltas).sum(axis=1)==0)[0]
    Stock_paths = Stock_paths[rows,:]
    deltas = deltas[rows,:]
    hedging_error_limit_bs, cost_limit_bs, hedging_portfolio_bs = new_evaluation.hedging_error_vector(strike, Stock_paths, option_price, deltas)
    hedging_error_limit_bs = -1*hedging_error_limit_bs

    #Consider same rows for deep hedging strategy    
    deep_hedging = deep_hedging[rows,:]

    hedging_error_limit, cost_limit, hedging_portfolio = new_evaluation.hedging_error_vector(strike, Stock_paths, option_price, deep_hedging)
    hedging_error_limit = -1*hedging_error_limit

    #Statistical arbitrage
    deltas_sa = deep_hedging-deltas
    strategy_value = np.zeros(Stock_paths.shape[0])
    for t in range(1,Stock_paths.shape[1]):
        strategy_value += deltas_sa[:,t-1]*(Stock_paths[:,t]*(discount_factor**t)*divided_factor-Stock_paths[:,t-1]*(discount_factor**(t-1)))   
    #strategy_value += option_price_1[0,0]
    strategy_value = strategy_value/(discount_factor**63)
    strategy_value_1 = -1*strategy_value
    risk_metric = np.mean(np.sort(strategy_value_1)[int(riskaversion*strategy_value_1.shape[0]):])

    #Targeted metrics
    cvar_bs = np.mean(np.sort(hedging_error_limit_bs)[int(riskaversion*hedging_error_limit_bs.shape[0]):])
    cvar_dh = np.mean(np.sort(hedging_error_limit)[int(riskaversion*hedging_error_limit.shape[0]):])

    #Compute average correlation of the hedging strategies across all time steps
    mean_value, df_results = regression_approach(deltas,deep_hedging)

    df_coef = df_results.iloc[:,[0,1,2,3,4,5]].copy()
    df_coef["Metrics"] = f"CVaR:{str(int(riskaversion*100))}%"
    df_coef["Model"] = model
    df_coefficients = pd.concat([df_coefficients, df_coef])

    df_summary_aux = pd.DataFrame()
    df_summary_aux["Strategy"] = [f"CVaR_{str(int(riskaversion*100))}"]
    df_summary_aux["Market"] = [model]
    df_summary_aux["Metric-CVaR"] = [cvar_dh]
    df_summary_aux["Relative_CVaR"] = [cvar_dh/cvar_bs]
    df_summary_aux["Avg correlation"] = [mean_value]
    df_summary_aux['Avg R-squared'] = [df_results['R-squared'].mean()]
    #df_summary_aux['Durbin-Watson'] = [df_results['Durbin-Watson'].mean()]
    #df_summary_aux['Breusch-Pagan'] = [df_results['Breusch-Pagan'].mean()]
    #df_summary_aux['Shapiro'] = [df_results['Shapiro'].mean()]
    df_summary_aux["Option_price"] = [option_price]
    df_summary_aux["CVaR_(DH-Delta)"] = [risk_metric]
    df_summary_aux["Mean_(DH-Delta)"] = [np.mean(strategy_value)]

    df_summary_risk = pd.concat([df_summary_risk, df_summary_aux])
    
    spread    = np.sqrt(((deltas-deep_hedging)**2).sum(axis=1))/np.sqrt(((deltas)**2).sum(axis=1))
    df_summary_aux = pd.DataFrame()
    df_summary_aux["Distance"] = spread
    df_summary_aux["Risk measure"] = f"CVaR:{str(int(riskaversion*100))}%"
    df_summary_aux["Market"] = model
    df_summary = pd.concat([df_summary, df_summary_aux])
    
    #change dir back to original working directory (owd)
    os.chdir(owd)

    return df_summary_risk

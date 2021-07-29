import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as scs
from scipy.optimize import minimize

def portfolio_moments(rets, lambda_factor):
    '''
    Input: Return array and lambda factor
    Output: annual return, volatility, Sharpe Ratio and Utility
    '''
    
    annual_mean = (np.prod(1+rets, axis=0))**(252/len(rets))-1
    annual_vola = np.std(rets, axis=0)*np.sqrt(252)
    
    sharpe_ratio = annual_mean / annual_vola
    utility = annual_mean - lambda_factor/2 * annual_vola**2
    
    return annual_mean, annual_vola, sharpe_ratio, utility

def exposure_mapping(df_mf_ret, g_ind, i_ind, lambda_factor):
    '''
    Input: 
        Macro Regime & Asset Return DataFrame
        Position of the Growth & Inflation Column in the DataFrame
        Lambad Factor for the utility calculation
    Output: 
        Portfolio Moments for each Regime
        Exposure Mapping for each Asset
        t-stats of the difference in average return
    '''
    #Define number of assets and get empty result arrays
    ast_numb = df_mf_ret.shape[1] - 2
    mus, stds = np.zeros((ast_numb,4)), np.zeros((ast_numb,4))
    srs, utlts = np.zeros((ast_numb,4)), np.zeros((ast_numb,4))
    
    #Define the four regimes based on the growth and inflation primer
    df_regime1 = df_mf_ret.loc[(df_mf_ret.iloc[:,g_ind]==1) & (df_mf_ret.iloc[:,i_ind]==1)]
    df_regime2 = df_mf_ret.loc[(df_mf_ret.iloc[:,g_ind]==1) & (df_mf_ret.iloc[:,i_ind]==0)]
    df_regime3 = df_mf_ret.loc[(df_mf_ret.iloc[:,g_ind]==0) & (df_mf_ret.iloc[:,i_ind]==1)]
    df_regime4 = df_mf_ret.loc[(df_mf_ret.iloc[:,g_ind]==0) & (df_mf_ret.iloc[:,i_ind]==0)]
    
    #Define the selection columns and get asset moments
    ast_cols = np.delete(list(range(ast_numb+2)), [g_ind, i_ind])
    mus[:,0], stds[:,0], srs[:,0], utlts[:,0] = portfolio_moments(df_regime1.iloc[:,ast_cols].values, lambda_factor)
    mus[:,1], stds[:,1], srs[:,1], utlts[:,1] = portfolio_moments(df_regime2.iloc[:,ast_cols].values, lambda_factor)
    mus[:,2], stds[:,2], srs[:,2], utlts[:,2] = portfolio_moments(df_regime3.iloc[:,ast_cols].values, lambda_factor)
    mus[:,3], stds[:,3], srs[:,3], utlts[:,3] = portfolio_moments(df_regime4.iloc[:,ast_cols].values, lambda_factor)
    
    #Caulculate the Growth and Inflation exposures
    grow_exp = np.mean(utlts[:,[0,1]], axis = 1) - np.mean(utlts[:,[2,3]], axis = 1)
    infl_exp = np.mean(utlts[:,[0,2]], axis = 1) - np.mean(utlts[:,[1,3]], axis = 1) 
    df_exp = pd.DataFrame([grow_exp, infl_exp], columns = df_mf_ret.columns[ast_cols])
    
    #Calculate the t-stats of difference in mean between the macro regimes
    gh_rets = df_mf_ret.loc[df_mf_ret.iloc[:,g_ind]==1].iloc[:,ast_cols]
    gl_rets = df_mf_ret.loc[df_mf_ret.iloc[:,g_ind]==0].iloc[:,ast_cols]
    ih_rets = df_mf_ret.loc[df_mf_ret.iloc[:,i_ind]==1].iloc[:,ast_cols]
    il_rets = df_mf_ret.loc[df_mf_ret.iloc[:,i_ind]==0].iloc[:,ast_cols]
    
    t_stats_g = scs.ttest_ind(gh_rets.values, gl_rets.values, 0)[0]
    t_stats_i = scs.ttest_ind(ih_rets.values, il_rets.values, 0)[0]
    
    return mus, stds, srs, utlts, df_exp, t_stats_g, t_stats_i

def asset_classifier_utlty(df_mf_ret, g_ind, i_ind, lambda_factor, K, opt):
    '''
    Input:
        Macro Regime & Asset Return DataFrame
        Position of the Growth & Inflation Column in the DataFrame
        Lambda Factor for the utility calculation
        K - Number of best assets per regime
        opt - if 1 gives asset names, 0 gives positions in DataFrame
    Output:
        The best assets for each regime
    '''
    ast_cols = np.delete(list(range(df_mf_ret.shape[1])), [g_ind, i_ind])
    ast_names = df_mf_ret.columns[ast_cols]    
    _,_,_,utilities,_,_,_ = exposure_mapping(df_mf_ret, g_ind, i_ind, lambda_factor)
    rank_pos = np.argsort(utilities, axis=0)
    ast_pos = np.reshape(rank_pos[-K:], K*4)
    asts = ast_names[rank_pos[-K:]]
    if opt == 1:
        return asts
    else:
        return ast_pos
    
def risk_parity_error(w, pars):
    '''
    Input:
        The asset weights (as array)
        Pars: The Covariance matrix & target Risk contribution (as array)
    Output:
        Error function
    '''
    w = np.matrix(w)
    cov_matrix = pars[0]
    w_t = np.matrix(pars[1])
    
    #Calculate the Risk Contribution of each asset under w
    sigma_w = np.sqrt(w@cov_matrix@w.T)
    marg_risk_contr = cov_matrix@w.T
    risk_contr = np.multiply(marg_risk_contr,w.T)/sigma_w.values
    
    #Get the error w/ squared error to risk-target
    risk_contr_t = np.asmatrix(np.multiply(w_t,sigma_w.values))
    error = sum(np.square(risk_contr.values - risk_contr_t.T))[0,0]
    
    return error*1000

def risk_parity_weights(w, covar_matr, w_t, constrtns):
    '''
    Input:
        The asset weights
        The Covariance matrix 
        Target Risk contribution
        Constraints: (0-none, 1-no leverage, 2-no shorting, 3-no leverage&short)
    Output:
        Risk-Parity Weights
    '''
    if constrtns == 0:
        cons = ()
    elif constrtns == 1:
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    elif constrtns == 2:
        cons = ({'type': 'ineq', 'fun': lambda w: w})
    elif constrtns == 3: 
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1}, {'type': 'ineq', 'fun': lambda w: w})
    #Do the optimization to get the optimal weights by minimizing error
    res = minimize(risk_parity_error, w, args=[covar_matr,w_t], method='SLSQP',constraints=cons, options={'disp': True})
    w_opt = np.asarray(res.x)
    
    return w_opt

def risk_parity_portfolio(df_mf_ret, g_ind, i_ind, lambda_factor, K, w, w_t, cnstr):
    '''
    Input:
        Macro Regime & Asset Returns DataFrame
        Position of the Growth & Inflation Column in the DataFrame
        Lambda Factor for the utility calculation
        K - Number of best assets per regime
        Starting asset weights
        Target Risk contribution
        Constraints: (0-none, 1-no leverage, 2-no shorting, 3-no leverage&short)
    Output:
        Risk-Parity Portfolio
        DataFrame of best assets per regime combination
        Risk Parity weights
    '''
    #Get the best assets and construct the best-asset DataFrame
    ast_pos = asset_classifier_utlty(df_mf_ret, g_ind, i_ind, lambda_factor, K, 0)
    ast_cols = np.delete(list(range(df_mf_ret.shape[1])), [g_ind, i_ind])
    df_ret_rp = df_mf_ret.iloc[:,ast_cols[ast_pos]]
    
    #Get the covariance matrix and optimal Risk-Parity weights
    cova_ast_rp = df_ret_rp.cov()
    w_opt = risk_parity_weights(w, cova_ast_rp, w_t, cnstr)
    rp_port = df_ret_rp.values@w_opt
    
    #Get moments of the resulting Risk-Parity portfolio
    mu_rp, vola_rp, sr_rp,_ = portfolio_moments(rp_port, 3.73)
    
    return rp_port, df_ret_rp, w_opt, mu_rp, vola_rp, sr_rp


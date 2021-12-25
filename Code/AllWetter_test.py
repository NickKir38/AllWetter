import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as scs
from scipy.optimize import minimize

#Import the macro factor data 
df_macro = pd.read_csv('GROWTH_INFLATION.csv')
df_regime = df_macro.iloc[:,[0,3,4]]
df_regime.datadate = pd.to_datetime(df_regime.datadate)

#Import return data for the indices and merge with the regime
df_returns = pd.read_csv('INDEX_RETURNS.csv')
df_returns.Date = pd.to_datetime(df_returns.Date)
df_regime.set_index('datadate', inplace = True)
df_returns.set_index('Date', inplace = True)
df_full = pd.merge(df_regime, df_returns, left_index = True, right_index = True)
df_full.dropna(inplace=True)

def covariance_shrinkage(df_returns, delta):
    '''
    Input:
        Return Dataframe
        Shrinkage factor
    Output:
        Shrunken covariance matrix
    '''
    #Get the sample covariance and define shrinkage matrix
    corr_s = df_returns.corr().values
    cov_s = df_returns.cov().values
    cov_f = np.full_like(cov_s, 0)
    
    #Fill the shrinkage matrix
    r_bar = 2/((len(cov_s)-1)*len(cov_s))*(np.triu(corr_s).sum() - np.trace(corr_s))
    for i in range(len(cov_f)):
        for j in range(len(cov_f)):
            if i == j:
                cov_f[i,j] = cov_s[i,j]
            else:
                cov_f[i,j] = r_bar * np.sqrt(cov_s[i,i] * cov_s[j,j]) 
    
    #Calculate the shrunken covariance matrix
    cov_shrink = delta * cov_s + (1-delta) * cov_f
    
    return cov_shrink

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
    df_exp = pd.DataFrame([grow_exp, infl_exp], 
                          columns = df_mf_ret.columns[ast_cols],
                          index = ['Growth', 'Inflation'])
    
    #Calculate the t-stats of difference in mean between the macro regimes
    gh_rets = df_mf_ret.loc[df_mf_ret.iloc[:,g_ind]==1].iloc[:,ast_cols]
    gl_rets = df_mf_ret.loc[df_mf_ret.iloc[:,g_ind]==0].iloc[:,ast_cols]
    ih_rets = df_mf_ret.loc[df_mf_ret.iloc[:,i_ind]==1].iloc[:,ast_cols]
    il_rets = df_mf_ret.loc[df_mf_ret.iloc[:,i_ind]==0].iloc[:,ast_cols]
    
    t_stats_g = scs.ttest_ind(gh_rets.values, gl_rets.values, 0)[0]
    t_stats_i = scs.ttest_ind(ih_rets.values, il_rets.values, 0)[0]
    
    return mus, stds, srs, utlts, df_exp, t_stats_g, t_stats_i

def asset_classifier_utlty(df_mf_ret, g_ind, i_ind, lambda_factor, K, opt=None):
    '''
    Input:
        Macro Regime & Asset Return DataFrame
        Position of the Growth & Inflation Column in the DataFrame
        Lambda Factor for the utility calculation
        K - Number of best assets per regime
        opt - if None gives asset names, else gives positions in DataFrame
    Output:
        The best assets for each regime
    '''
    ast_cols = np.delete(list(range(df_mf_ret.shape[1])), [g_ind, i_ind])
    ast_names = df_mf_ret.columns[ast_cols]    
    _,_,_,utlts,_,_,_ = exposure_mapping(df_mf_ret, g_ind, i_ind, lambda_factor)
    rank_pos = np.argsort(utlts, axis=0)
    ast_pos = np.reshape(rank_pos[-K:], K*4)
    asts = ast_names[rank_pos[-K:]]
    if opt == None:
        return pd.DataFrame(asts)
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
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1}, 
                {'type': 'ineq', 'fun': lambda w: w})
    #Do the optimization to get the optimal weights by minimizing error
    res = minimize(risk_parity_error, w, args=[covar_matr,w_t], 
                   method='SLSQP', constraints=cons, options={'disp': True})
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
    sel_asts = ast_cols[ast_pos]
    df_ret_rp = df_mf_ret.iloc[:,sel_asts ]
    
    #Get the covariance matrix and optimal Risk-Parity weights
    cova_ast_rp = df_ret_rp.cov()
    w_opt = risk_parity_weights(w, cova_ast_rp, w_t, cnstr)
    rp_port = df_ret_rp.values@w_opt
    
    #Get moments of the resulting Risk-Parity portfolio
    mu_rp, vola_rp, sr_rp,_ = portfolio_moments(rp_port, lambda_factor)
    rp_moms = [mu_rp, vola_rp, sr_rp]
    df_w = pd.DataFrame(w_opt, index = df_ret_rp.columns)
    
    return rp_port, df_ret_rp, df_w, rp_moms, sel_asts
    
def backtester(df_mf_ret, g_ind, i_ind, lambda_factor, K, w, w_t, cnstr, q, L):
    '''
    Input:
        Macro Regime & Asset Returns DataFrame
        Position of the Growth & Inflation Column in the DataFrame
        Lambda Factor for the utility calculation
        K - Number of best assets per regime
        Starting asset weights
        Target Risk contribution
        Constraints: (0-none, 1-no leverage, 2-no shorting, 3-no leverage&short)
        q (0<q<1): Percentage used for training, 1-q is used for out-of-sample 
        L: Invested amount at the beginning
    Output:
        in-sample returns
        out-of-sample returns
        return moments (IS & OOS)
        Amount made OOS
    '''
    #Select the in-sample data for getting the weights
    if q == None: 
        odd_day = df_mf_ret.index.day%2
        df_in_sample = df_mf_ret.iloc[odd_day==1,:]
        df_out_sample = df_mf_ret.iloc[odd_day==0,:]
        
    else:
        len_df = df_mf_ret.shape[0]
        in_sample = round(q*len_df)
        df_in_sample = df_mf_ret.iloc[0:in_sample,:]
        df_out_sample = df_mf_ret.iloc[in_sample+1:len_df,:]
    
    #Fit the insample data to get the optimal weights
    rp_ret_is, _, w_opt, _,sel_asts = risk_parity_portfolio(df_in_sample, g_ind, i_ind, lambda_factor, K, w, w_t, cnstr)
    df_out_sample = df_out_sample.iloc[:,sel_asts]
    rp_ret_os = df_out_sample.values@w_opt
    
    #Fit the currency at the beginning (no rebalancing)
    inv_amnts = w_opt * L
    rp_ret_blncs = np.cumprod(df_out_sample+1)@inv_amnts
    
    #Create dataframes
    df_rp_ret_is = pd.DataFrame(rp_ret_is, index = df_in_sample.index)
    df_rp_ret_os = pd.DataFrame(rp_ret_os.values, index = df_out_sample.index)
    
    #Get the portfolio moments
    mu_is, vola_is, sr_is, _ = portfolio_moments(rp_ret_is, lambda_factor)
    mu_os, vola_os, sr_os, _ = portfolio_moments(rp_ret_os.values, lambda_factor)
    is_moms = np.asarray([mu_is, vola_is, sr_is])
    os_moms = np.asarray([mu_os, vola_os, sr_os])
    
    return df_rp_ret_is, df_rp_ret_os, is_moms, os_moms, rp_ret_blncs

def drawdown(returns):
    '''
    Input:
        Return series
    Output:
        Array of Drawdown sequence and a plot of it
    '''
    cum_ret = np.cumprod(returns+1)
    drawdown_array = cum_ret / np.maximum.accumulate(cum_ret) - 1
    plt.plot(drawdown_array)
    plt.show()
    
    return drawdown_array

#Get the epxosures
_, _, _, _, df_exp, t_stats_g, t_stats_i = exposure_mapping(df_full, 0, 1, 0.1)

#Define number of assets per regime & starting/risk parity weights
K = 3
w = [0.33]*(K*4)
w_T = [0.33]*(K*4)
rp_port, df_ret_rp, df_w, rp_moms, _ = risk_parity_portfolio(df_full, 0, 1, 3.73, K, w, w_T, 3)

#Backtesting with OOS
df_ret_is, df_ret_os, mom_is, mom_os, _ = backtester(df_full, 0, 1, 3.73, 3, w, w_T, 3, None, 1000)
dd_os = drawdown(df_ret_os)
plt.plot(np.cumprod(df_ret_os+1))
plt.show()
# AllWetter

This Python library includes functions to construct all-weather portfolios from a set of asset returns based on their performance during different macroeconomic regimes.

![first equation](https://latex.codecogs.com/gif.latex?a%20%3D%20%5Cfrac%7B2%7D%7B3%7D)

## Usage

The reposertory includes a CSV file that includes the time series of the forward-looking CLI index and Breakeven inflation. These two measure market expectations on Growth and Inflation. Further, the table includes for both a 0 and 1 to indicate whether the said expectation is rising or falling.

Based on these, four macro-economic regimes are formulated. Recession, Goldilock, Stagflation and Expansion each represent the possible combinations of rising/falling Growth and Inflation expectations.

Then, one can rank the K best assets during each regime and create a risk-parity porftolio based on the regimes. 

## Risk-Parity

The goal of risk-parity portfolios is to have each included asset of the portfolio to contribute the same amount of risk in form of marginal volatility as the others to the portfolio. This can be defined as:

![first equation](https://latex.codecogs.com/gif.latex?%5Csigma_p%20%3D%20%5Csqrt%7Bw%5COmega%20w%27%7D)
![first equation](https://latex.codecogs.com/gif.latex?RC_j%20%3D%20w_j%20%5Ctimes%20MRC%20%3D%20w_j%20%5Ctimes%20%5Cfrac%7B%5Cdelta%20%5Csigma_p%7D%7B%5Cdelta%20w_j%7D%20%3D%20w_j%5Ctimes%5Cfrac%7B%5COmega%5Ctimes%20w%7D%7B%5Csigma_p%7D)

In this denotation $\sigma_p$ is the risk (volatility) of the portfolio, $\Omega$ the covariance matrix of the assets and $w$ the vector of asset weights. This way we can obtain the Risk Contribution of each asset to the portfolio's risk and choose asset weights in a way to obtain an (equal) risk contribution.

To do this we minimize:
$$\epsilon(w) = \sum_{i=1}^{n} (RC_i-w_{t,i} \times \sigma_w)^2$$
After defining a target level of risk contribtution for each asset $w_{t}$ we use that to get the error against the chosen input weight $w$ to get the optimum weights $\textbf{w}$ that achieve the targeted risk contribution.

## Further functionalities

Besides obtaining the risk-based portfolio of the K best assets in each regime based on a targeted risk contribution one can also run backtests, comparing the resulting portfolio in-sample over $q%$ of the data set. Also, a function to calculate the portfolio moments of a return series is included. Based on a returns series' resulting ultility $U=r_p - \frac{\lambda}{2} \sigma_p^2$ there is also an assessment and plotting of each assets' macro exposure possible.

Missing: Drawdown function.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ############################################################
# You need to replace the ??? characters by actual code
# ############################################################


# ############################################################
# 1 - Data                                                  #
# ############################################################

# Define the main path
mainpath = r"C:/Users/Emile/Desktop/cours/ESILV/A4/S2/SPECIALISATION_IF/ML&AM/TD1/"
# Load the data
all_assets_prices = pd.read_csv(f"{mainpath}DataForStatsTutorial1.csv", index_col="Dates", sep=";", parse_dates=["Dates"], dayfirst=True)
#all_assets_prices = pd.read_csv("DataForStatsTutorial1.csv", index_col=0, sep=";", parse_dates=True, dayfirst=True)

# ############################################################
# 2 - Data Exploration                                       #
# ############################################################

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(all_assets_prices[['DMEquitiesEUR', 'DMEquitiesUSD']])
plt.yscale("log")
plt.title("ES50 and SP500")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(['DMEquitiesEUR', 'DMEquitiesUSD'])
plt.show()

# Plot a subset of the data
subset = all_assets_prices.loc['2019':'2022', ['DMEquitiesEUR', 'DMEquitiesUSD']]
plt.figure(figsize=(10, 5))
plt.plot(subset)
plt.title("ES50 and SP500 - From 2019 to 2022")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(['DMEquitiesEUR', 'DMEquitiesUSD'])
plt.show()

# Change frequency
all_assets_prices_daily = all_assets_prices
all_assets_prices_weekly = all_assets_prices.resample('W').last()
all_assets_prices_monthly = all_assets_prices.resample('M').last()

# Display the first few rows of each DataFrame
print("Daily Prices Head:")
print(all_assets_prices_daily.head())
print("\nWeekly Prices Head:")
print(all_assets_prices_weekly.head())
print("\nMonthly Prices Head:")
print(all_assets_prices_monthly.head())

# Function to compute returns from prices
def compute_return(price):
    ret = price / price.shift(1) - 1
    return ret

# Example: Calculate returns for SP500
prices_sp = all_assets_prices_daily['DMEquitiesUSD']
returns_sp = compute_return(prices_sp)

# Compare with the pct_change function
returns_sp_check = prices_sp.pct_change()

# Display the first few returns
print("\nReturns SP Head:")
print(returns_sp.head())


#######################################
# 3 - Usual stats                     #
#######################################

# Function to compute Compound Annual Growth Rate (CAGR)
def compute_cagr(price, ann_multiple=252):
    n = len(price)
    cagr = (price.iloc[-1] / price.iloc[0]) ** (ann_multiple / n) - 1
    return cagr

# Example: Calculate CAGR for SP500
cagr_sp = compute_cagr(all_assets_prices['DMEquitiesUSD'])
print(f'CAGR of SP 500 is: {round(cagr_sp * 100, 2)}%')

# Function to compute Volatility
def compute_vol(price, ann_multiple=252):
    ret = price / price.shift(-1) - 1  # Calculate returns
    n = len(price)  # Get the length of the time series
    mu = ret.mean()  # Calculate the mean
    sigma_daily = np.sqrt((ret - mu).pow(2).sum() / (n - 1))  # Calculate daily volatility
    sigma = np.sqrt(252) * sigma_daily  # Annualize the volatility
    return sigma

# Example: Calculate Volatility for SP500
vol_sp = compute_vol(all_assets_prices['DMEquitiesUSD'])
print(f'Volatility of SP 500 is: {round(vol_sp * 100, 2)}%')

# Comparing with the std function
sp_returns = all_assets_prices['DMEquitiesUSD'].pct_change()
vol_sp_check = np.sqrt(252)*all_assets_prices['DMEquitiesUSD'].pct_change().std()

# Function to compute Drawdown
def compute_dd_np(price):
    price = price.values  # Convert to numpy array
    drawdown = price / np.maximum.accumulate(price) - 1  # Calculate drawdown
    return drawdown

def compute_dd_pd(df,series):
    # Cumulative returns
    series_to_retain = df[series]
    out_df = pd.DataFrame()
    out_df[series] = series_to_retain
    out_df['Cumulative'] = out_df / out_df.iloc[0]
    # Maximum value up to each point
    out_df['Max'] = out_df['Cumulative'].cummax()
    # DD
    out_df['Drawdown'] = (out_df['Cumulative'] - out_df['Max']) / out_df['Max']
    return out_df

# Example: Calculate Drawdown for SP500
dd_np = compute_dd_np(all_assets_prices['DMEquitiesUSD'])  # Calculate drawdowns with numpy
dd_df = compute_dd_pd(all_assets_prices,'DMEquitiesUSD')  # Calculate drawdowns wih pandas

plt.figure(figsize=(10, 5))
plt.plot(dd_df.Drawdown, linewidth=2)
plt.title("SP500 Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.show()

mdd = np.min(dd_np)  # Max drawdown
print(f"Max Drawdown of SP 500 = {round(mdd * 100, 2)}%")
print(f"Max Drawdown of SP 500 = {round(dd_df.Drawdown.min() * 100, 2)}%")

# Function to compute Sharpe Ratio
def compute_sr(price, ret_without_risk=0.00, ann_multiple=252):
    cagr = compute_cagr(price, ann_multiple)  # CAGR
    vol = compute_vol(price, ann_multiple)  # Volatility
    sr = (cagr - ret_without_risk) / vol
    return sr

# Example: Calculate Sharpe Ratio for SP500
rf = 0.02  # Risk-free rate
sr = compute_sr(all_assets_prices['DMEquitiesUSD'], rf, 252)  # Sharpe Ratio
print(f"Sharpe Ratio of SP 500 = {round(sr, 2)}")

# Examples to get subsets
prix_sp_subset1 = all_assets_prices.loc["2001-01-01":"2003-12-31", "DMEquitiesUSD"]
prix_sp_subset2 = all_assets_prices.loc["2001-04-01":"2004-03-31", "DMEquitiesUSD"]
prix_sp_subset3 = all_assets_prices.loc["2001-07-01":"2004-06-30", "DMEquitiesUSD"] 

sr_1 = compute_sr(prix_sp_subset1, rf, 252)  # Sharpe Ratio 2000-2007
print(f"Sharpe Ratio of SP500 Index (2000-2007) = {round(sr_1, 2)}")

sr_2 = compute_sr(prix_sp_subset2, rf, 252)  # Sharpe Ratio 2008-2009
print(f"Sharpe Ratio of SP500 Index (2008-2009) = {round(sr_2, 2)}")

sr_3 = compute_sr(prix_sp_subset3, rf, 252)  # Sharpe Ratio 2010-2022
print(f"Sharpe Ratio of SP500 Index (2010-2022) = {round(sr_3, 2)}")

# Rolling Sharpe Ratio
rolling_sr = all_assets_prices['DMEquitiesUSD'].rolling(window=252).apply(compute_sr, raw=False)
plt.figure(figsize=(10, 5))
plt.plot(rolling_sr)
plt.title("Rolling Sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.show()

# Expanding Sharpe Ratio
expanding_sr = all_assets_prices['DMEquitiesUSD'].expanding().apply(compute_sr, raw=False)
expanding_sr[~np.isfinite(expanding_sr)] = 0  # Replace non-finite values with 0
plt.figure(figsize=(10, 5))
plt.plot(expanding_sr)
plt.title("Expanding Sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.show()

# Alternative way to calculate expanding Sharpe Ratio wiht lambda
expanding_sr = all_assets_prices['DMEquitiesUSD'].expanding().apply(lambda x: compute_sr(x), raw=False)

# Replace non-finite values with 0
expanding_sr[~np.isfinite(expanding_sr)] = 0

# Plotting the Expanding Sharpe Ratio
plt.figure(figsize=(10, 5))
plt.plot(expanding_sr)
plt.title("Expanding Sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.show()

# #############################################################
# 4 - Calculation of Covariances and Correlations            #
# #############################################################

# Get data
eight_assets_prices_daily = all_assets_prices_daily[['DMEquitiesEUR', 'DMEquitiesUSD',
                                                      'BondsDEM', 'BondsGBP',
                                                      'BondsUSD', 'DMFXCHF',
                                                      'DMFXEUR', 'DMFXGBP']]

# Calculate returns
returns = eight_assets_prices_daily.apply(???)  # Apply: Loop for columns
returns = returns.dropna()  # Remove rows that contain NaN

# Covariance calculation
n = returns.shape[0]  # Number of rows
Mc = ??? - ???  # Center the data: M - mean(M)
S = (??? @ ???) / (n - 1)  # Classic formula

S_check = eight_assets_prices_daily.pct_change().cov()

def cov_mtx(ret, ann_multiple=252):
    cmtx = ret.cov() * ann_multiple
    return cmtx

cov_matrix = cov_mtx(returns)
print("Covariance Matrix:")
print(cov_matrix)

# Verification
print("Checking Covariance Matrix:")
print(cov_mtx(returns) - S * 252)

# Correlation calculation
Ms = (??? - ???)) / ???  # Centered and scaled data
R = (??? @ ???) / (n - 1)  # Also see Wikipedia

def corr_mtx(ret):
    cmtx = ret.corr()
    return cmtx

corr_matrix = corr_mtx(returns)
print("Correlation Matrix:")
print(corr_matrix)

# Verification
print("Checking Correlation Matrix:")
print(corr_mtx(returns) - R)

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix")
plt.show()

# Correlations on other historical data
# Example for the period 2000-2007
returns_subset1 = eight_assets_prices_daily.???[???:???].apply(???)
rho1 = corr_mtx(returns_subset1)

plt.figure(figsize=(10, 8))
sns.heatmap(rho1, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix (2000-2007)")
plt.show()

# Example for the period 2008-2009
returns_subset2 = eight_assets_prices_daily.???[???:???].apply(compute_return)
rho2 = corr_mtx(returns_subset2)

plt.figure(figsize=(10, 8))
sns.heatmap(rho2, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix (2008-2009)")
plt.show()

# Example for the period 2010-2022
returns_subset3 = eight_assets_prices_daily.???[???:???].apply(compute_return)
rho3 = corr_mtx(returns_subset3)

plt.figure(figsize=(10, 8))
sns.heatmap(rho3, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix (2010-2022)")
plt.show()

# #############################################################
# 5 - Building Indices                                     #
# #############################################################

# 5.1 Indices without volatility constraint
# Get data (Multiasset)
prices_multi_asset_daily = eight_assets_prices_daily

# Get data (Equity)
prices_equity_daily = all_assets_prices_daily[['DMEquitiesCAD', 'DMEquitiesCHF',
                                               'DMEquitiesDEM', 'DMEquitiesFRF',
                                               'DMEquitiesGBP', 'DMEquitiesJPY',
                                               'DMEquitiesNDQ', 'DMEquitiesUSD']]

# Calculate returns
return_multi_asset = prices_multi_asset_daily.apply(compute_return)
return_equity = prices_equity_daily.apply(compute_return)

# Calculate correlations for equities
rho_multi_asset = return_multi_asset.corr()
rho_equities = return_equity.corr()

# Plot correlation matrix for equities
plt.figure(figsize=(10, 8))
sns.???(rho_equities, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix for Equities")
plt.show()

# Mean correlations
mean_rho_equities = rho_equities.values[np.???(rho_equities, k=1)].mean()
mean_rho_multi_asset = rho_multi_asset.values[np.???(rho_multi_asset, k=1)].mean()
print(f"Mean correlation for equities: {mean_rho_equities}")
print(f"Mean correlation for multi-assets: {mean_rho_multi_asset}")

# Monthly correlations
return_equities_monthly = prices_equity_daily.???(???).apply(compute_return)
rho_equities_monthly = return_equities_monthly.corr()
mean_rho_equities_monthly = rho_equities_monthly.values[np.???(rho_equities_monthly, k=1)].mean()
print(f"Mean correlation for equities (monthly): {mean_rho_equities_monthly}")

# Rebalancing dates (quarterly)
rebalancing_dates = prices_multi_asset_daily.???(???).last().index

# Initialize strategy returns as DataFrames
strategy_ret_ew_ma = pd.DataFrame(index=return_multi_asset.index)
strategy_ret_ew_ma['ew_ma'] = None
strategy_ret_ew_eq = pd.DataFrame(index=return_equity.index)
strategy_ret_ew_eq['ew_eq'] = None
strategy_ret_vp_ma = pd.DataFrame(index=return_multi_asset.index)
strategy_ret_vp_ma['vp_ma'] = None
strategy_ret_vp_eq = pd.DataFrame(index=return_equity.index)
strategy_ret_vp_eq['vp_eq'] = None

equal_weight_ma = np.repeat(1 / return_multi_asset.shape[1], return_multi_asset.shape[1])
equal_weight_eq = np.repeat(1 / return_equity.shape[1], return_equity.shape[1])

for index_t in range(len(rebalancing_dates) - 1):
    t = ???[???]

    # Equal weight portfolios
    weight_ew_ma = equal_weight_ma
    weight_ew_eq = equal_weight_eq

    # Data for the past year
    data_one_year_multi_asset = return_multi_asset.loc[???:???]
    data_one_year_equity = return_equity.loc[???:???]

    # Calculate covariance matrices and retain volatilities
    vol_ma = np.sqrt(np.???(???(???, 252)))
    vol_eq = np.sqrt(np.???(???(???, 252)))

    # Calculate weights for volatility parity
    weight_vp_ma = equal_weight_ma / vol_ma
    weight_vp_ma /= weight_vp_ma.sum()
    weight_vp_eq = equal_weight_eq / vol_eq
    weight_vp_eq /= weight_vp_eq.sum()

    # Rebalancing
    index_return = slice(rebalancing_dates[index_t] + pd.Timedelta(days=1),
                         rebalancing_dates[index_t + 1])

    strategy_ret_ew_ma.loc[index_return,'ew_ma'] = return_multi_asset.loc[???].dot(???)
    strategy_ret_ew_eq.loc[index_return,'ew_eq'] = return_equity.loc[???].dot(???)
    strategy_ret_vp_ma.loc[index_return,'vp_ma'] = return_multi_asset.loc[???].dot(???)
    strategy_ret_vp_eq.loc[index_return,'vp_eq'] = return_equity.loc[???].dot(???)

# Backtest
strategy_ret_recap = strategy_ret_ew_ma.join(strategy_ret_ew_eq.join(strategy_ret_vp_ma.join(strategy_ret_vp_eq)))
strategy_ret_recap.fillna(0, inplace=True)
strategy_cumret_recap = (1+strategy_ret_recap).cumprod()

# Plotting the strategies
plt.figure(figsize=(10, 5))
plt.plot(strategy_cumret_recap)
plt.title("Initial Strategies")
plt.legend(strategy_ret_recap.columns)
plt.show()

# Print Sharpe Ratios
print('Sharpe Ratio of 4 portfolios:')
print(strategy_cumret_recap.apply(compute_sr))

# Print Volatility of 4 portfolios
print('Volatility of 4 portfolios:')
print(strategy_cumret_recap.apply(compute_vol))

# Print Sharpe Ratio of equities
print('Sharpe Ratio of equities:')
print(prices_equity_daily.apply(compute_sr))

# Print Sharpe Ratio of assets in multi-asset portfolios
print('Sharpe Ratio of assets in multi-asset portfolios:')
print(prices_multi_asset_daily.apply(compute_sr))

# 5.2 Indices with target volatility
# Target Volatility
TARGVOL = 0.1

# Initialize strategy returns as DataFrames
strategy_ret_ew_ma_tv = pd.DataFrame(index=return_multi_asset.index)
strategy_ret_ew_ma_tv['ew_ma'] = None
strategy_ret_ew_eq_tv = pd.DataFrame(index=return_equity.index)
strategy_ret_ew_eq_tv['ew_eq'] = None
strategy_ret_vp_ma_tv = pd.DataFrame(index=return_multi_asset.index)
strategy_ret_vp_ma_tv['vp_ma'] = None
strategy_ret_vp_eq_tv = pd.DataFrame(index=return_equity.index)
strategy_ret_vp_eq_tv['vp_eq'] = None

# Equal weights
equal_weight_ma = np.repeat(1 / return_multi_asset.shape[1], return_multi_asset.shape[1])
equal_weight_eq = np.repeat(1 / return_equity.shape[1], return_equity.shape[1])

# Rebalancing dates (assuming you have a function to get these)
rebalancing_dates = prices_multi_asset_daily.resample('QE').last().index

for index_t in range(len(rebalancing_dates) - 1):
    t = rebalancing_dates[index_t]

    # Data for the past year
    data_one_year_multi_asset = return_multi_asset.loc[???:???]
    data_one_year_equity = return_equity.loc[???:???]

    # Compute ex ante vol
    vcv_ma = ???(data_one_year_multi_asset, 252).values
    vcv_eq = ???(data_one_year_equity, 252).values

    # Calculate volatilities using numpy
    vol_ew_ma = np.sqrt(np.???(equal_weight_ma, np.???(vcv_ma, equal_weight_ma)))
    vol_ew_eq = np.sqrt(np.???(equal_weight_eq, np.???(vcv_eq, equal_weight_eq)))

    # Compute leverage
    lev_ew_ma = TARGVOL / ???
    lev_ew_eq = TARGVOL / ???

    # Adjust weights for target volatility
    weight_ew_ma_tv = equal_weight_ma * lev_ew_ma
    weight_ew_eq_tv = equal_weight_eq * lev_ew_eq

    # Compute covariance matrix for volatility parity
    vol_ma = np.sqrt(np.???(???(???, 252).values))
    vol_eq = np.sqrt(np.???(???(???, 252).values))

    weight_vp_ma = equal_weight_ma / vol_ma
    weight_vp_ma /= weight_vp_ma.sum()
    weight_vp_eq = equal_weight_eq / vol_eq
    weight_vp_eq /= weight_vp_eq.sum()

    vol_vp_ma = np.sqrt(np.???(weight_vp_ma, np.???(vcv_ma, weight_vp_ma)))
    vol_vp_eq = np.sqrt(np.???(weight_vp_eq, np.???(vcv_eq, weight_vp_eq)))

    # Compute leverage for volatility parity
    lev_vp_ma = TARGVOL / vol_vp_ma
    lev_vp_eq = TARGVOL / vol_vp_eq

    weight_vp_ma_tv = weight_vp_ma * lev_vp_ma
    weight_vp_eq_tv = weight_vp_eq * lev_vp_eq

    # Assign returns for the index range
    # Rebalancing
    index_return = slice(rebalancing_dates[index_t] + pd.Timedelta(days=1),
                         rebalancing_dates[index_t + 1])

    strategy_ret_ew_ma_tv.loc[index_return, 'ew_ma'] = return_multi_asset.loc[???].dot(???)
    strategy_ret_ew_eq_tv.loc[index_return, 'ew_eq'] = return_equity.loc[???].dot(???)
    strategy_ret_vp_ma_tv.loc[index_return, 'vp_ma'] = return_multi_asset.loc[???].dot(???)
    strategy_ret_vp_eq_tv.loc[index_return, 'vp_eq'] = return_equity.loc[???].dot(???)

# Backtest
strategy_ret_recap_tv = strategy_ret_ew_ma_tv.join(strategy_ret_ew_eq_tv.join(strategy_ret_vp_ma_tv.join(strategy_ret_vp_eq_tv)))
strategy_ret_recap_tv.fillna(0, inplace=True)
strategy_cumret_recap_tv = (1+strategy_ret_recap_tv).cumprod()

# Plotting the strategies
plt.figure(figsize=(10, 5))
plt.plot(strategy_cumret_recap_tv)
plt.title("Initial Strategies")
plt.legend(strategy_ret_recap_tv.columns)
plt.show()

# Print Sharpe Ratios
print('Sharpe Ratio of 4 portfolios:')
print(strategy_cumret_recap_tv.apply(compute_sr))

# Print Volatility of 4 portfolios
print('Volatility of 4 portfolios:')
print(strategy_cumret_recap_tv.apply(compute_vol))

# Print Sharpe Ratio of equities
print('Sharpe Ratio of equities:')
print(prices_equity_daily.apply(compute_sr))

# Print Sharpe Ratio of assets in multi-asset portfolios
print('Sharpe Ratio of assets in multi-asset portfolios:')
print(prices_multi_asset_daily.apply(compute_sr))


# ############################################################
# 6 - A first model                                          #
# ############################################################

# Read the CSV file
model_assets_prices = pd.read_csv(f"{mainpath}DataForModelTutorial1.csv", index_col=???, sep=???, parse_dates=[???], dayfirst=True)

# Transformations
model_assets_prices_for_model = model_assets_prices.copy()
model_assets_prices_for_model['CorePCE'] = ???
model_assets_prices_for_model['Unemployment'] = ???
model_assets_prices_for_model['SOFRSpread'] =???

# Prepare inputs for OLS
inputs = pd.DataFrame({
    'CorePCE': model_assets_prices_for_model['CorePCE'],
    'Unemployment': model_assets_prices_for_model['Unemployment'],
    'SOFRSpread': model_assets_prices_for_model['SOFRSpread'],
    'FedFunds': model_assets_prices_for_model['FedFunds']
})

# Full history OLS regression
data_for_ols = pd.DataFrame({'y': model_assets_prices_for_model['US10Y'], **inputs})
X = sm.add_constant(data_for_ols.drop(columns='y'))  # Add constant for intercept
lin_model_full = sm.???(data_for_ols['y'], ???).???()

# Summary of the full model
print(lin_model_full.summary())

# Predictions
data_for_ols['US10Yhat'] = lin_model_full.???(X)

# Quick and dirty chart
plt.figure(figsize=(10, 5))
plt.plot(data_for_ols.index, data_for_ols['y'], label='Actual US10Y', color='blue')
plt.plot(data_for_ols.index, data_for_ols['US10Yhat'], label='Predicted US10Y', color='red')
plt.title('US10Y Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('US10Y')
plt.legend()
plt.show()

# Last 10 years OLS regression
last_10_years = data_for_ols.tail(???)  # Assuming the DataFrame is sorted by date
X_last_10y = sm.add_constant(last_10_years.drop(columns=['y','US10Yhat']))
lin_model_last_10y = sm.???(last_10_years['y'], ???).???()

# Summary of the last 10 years model
print(lin_model_last_10y.summary())

# Last 5 years OLS regression
last_5_years = data_for_ols.tail(???)
X_last_5y = sm.add_constant(last_5_years.drop(columns=['y','US10Yhat']))
lin_model_last_5y = sm.???(last_5_years['y'], ???).???()

# Summary of the last 5 years model
print(lin_model_last_5y.summary())
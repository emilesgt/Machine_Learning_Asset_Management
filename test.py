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
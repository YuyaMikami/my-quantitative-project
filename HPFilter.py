
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_datareader as pdr
import numpy as np

# set the start and end dates for the data
start_date = '1955-01-01'
end_date = '2022-01-01'

# download the data from FRED using pandas_datareader
gdp = web.DataReader('NGDPRSAXDCKRQ', 'fred', start_date, end_date)
log_gdp = np.log(gdp)

# apply a Hodrick-Prescott filter to the data to extract the cyclical component
cycle10, trend10 = sm.tsa.filters.hpfilter(log_gdp, lamb=10)
cycle100, trend100 = sm.tsa.filters.hpfilter(log_gdp, lamb=100)
cycle1600, trend1600 = sm.tsa.filters.hpfilter(log_gdp, lamb=1600)

# Plot the original time series data
plt.plot(log_gdp, label="Original GDP (in log)")

# Plot the trend component
plt.plot(trend10, label="Trend(λ=10)")
plt.plot(trend100, label="Trend(λ=100)")
plt.plot(trend1600, label="Trend(λ=1600)")

plt.title("GDP trend")
plt.xlabel("Year")
plt.ylabel("Log GDP")

plt.grid(True)
# Add a legend and show the plot
plt.legend()
plt.show()

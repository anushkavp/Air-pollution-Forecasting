from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
import statsmodels.tsa.holtwinters as ets

pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL, seasonal_decompose

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore",InterpolationWarning)


from toolset import *
np.random.seed(6313)

df = pd.read_csv("Beijing_air_pollution.csv",index_col="date")
df.index = pd.to_datetime(df.index)
print("Daatframe shape: "+str(df.shape))
print("\n\nDataset Description")
print(df.info())

print("\n\nSample of dataframe:")
print(df.head())

################
# Description of Dataset
################
print("Checking for missing values:")
print(df.isna().sum())

print("Total number of rows:",str(len(df)))

# One hot encoding:
print("\n\nOne hot encoding")
print("There is one column that is categorical in nature. One hot encoding:")
df = pd.get_dummies(df,columns=["wnd_dir"])
print(df.head())

# Numerical stats of dataframe:
print("\n\nNumerical Stats on the dataset:")
print(df.describe())

# Time series plot
plt.figure()
df["pollution"].plot()
plt.xlabel("Date")
plt.ylabel("Pollution Levels")
plt.title("Time Series plot of Air Pollution levels")
plt.grid()
plt.legend()
plt.show()

# ACF plot of pollution levels
Cal_autocorrelation(df["pollution"],20,"ACF function of Beijing Air pollution")
# Cal_autocorrelation(df["pollution"],100,"ACF function of Beijing Air pollution")

# Correlation matrix
plt.figure(figsize=(12, 10))
ax=sns.heatmap(df.corr(), annot=True,vmin=-1,vmax=1, center=0, cmap = sns.diverging_palette(20,220,n=200), square=True)
bottom,top = ax.get_ylim()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
plt.title("Heatmap plot for Air Pollution dataset")
plt.show()

# splitting into train and test
print("\n\nSplitting data into train and test")
y=df["pollution"]
x=df.loc[:, df.columns != 'pollution']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6313, shuffle= False)
print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)
print("x_test shape: ",x_test.shape)
print("y_test shape: ",y_test.shape)


################
# Stationarity
################
# print("\n\n\nStationarity Test:")
# # Plot rolling mean and variance of Air pollution
# Plot_Rolling_Mean_Var(df,"pollution")
#
# # ADF test for original dataset
# print("\nADF Test for Beijing Air Pollution dataset:")
# ADF_Cal(df.loc[:,"pollution"])
#
# # KPSS test for original dataset
# print("\nKPSS Test for Beijing Air Pollution dataset:")
# kpss_test(df.loc[:,"pollution"])
#
# # Applying first order differencing
# df["First Order Differencing"]=0
# for i in range(1,len(df)):
#     # First Order Differencing
#     df["First Order Differencing"].iloc[i]= (
#             df["pollution"].iloc[i] -
#             df["pollution"].iloc[i-1])
#
# df.to_csv("df_stationarized.csv")
# df_stationarized = pd.read_csv("df_stationarized.csv")
#
# # Plot rolling mean and variance of first order differenced dataset
# Plot_Rolling_Mean_Var(df_stationarized, "First Order Differencing")
#
# # ADF test
# print("\nADF Test for First Order transformation:")
# ADF_Cal(df.loc[:,"First Order Differencing"])
#
# # KPSS test
# print("\nKPSS Test for First Order transformation:")
# kpss_test(df.loc[:,"First Order Differencing"])
#
# Cal_autocorrelation(df["pollution"],20,"ACF function of Beijing Air pollution")
# Cal_autocorrelation(df["First Order Differencing"],20,"ACF function of First Order Differencing of air pollution")

################
# Time Series Decomposition
################
print("\n\nTime Series Decomposition")
stl = STL(df["pollution"])
res = stl.fit()

fig = res.plot()
fig.set_size_inches(12, 10)
plt.xlabel("Date")
fig.suptitle("Air Pollution Time Series Decomposition")
plt.show()


T = res.trend
S = res.seasonal
R=res.resid

plt.figure(figsize=(12,10))
plt.plot(df["pollution"], label="Original air pollution")
plt.plot(T, label="Trend")
plt.plot(S, label="Seasonality")
plt.plot(R, label="Residuals")
plt.xlabel("Date")
plt.ylabel("Air Pollution PM2.5 levels")
plt.title("Air Pollution Time Series Decomposition for all values")
plt.xticks(rotation = 45)
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12,10))
plt.plot(df["pollution"].iloc[:50], label="Original air pollution")
plt.plot(T[:50], label="Trend")
plt.plot(S[:50], label="Seasonality")
plt.plot(R[:50], label="Residuals")
plt.xlabel("Date")
plt.ylabel("Air Pollution PM2.5 levels")
plt.title("Air Pollution Time Series Decomposition for first 50 values")
plt.xticks(rotation = 45)
plt.legend()
plt.grid()
plt.show()

seasonally_adjusted_data = T+R
plt.figure(figsize=(12,9))
plt.plot(df["pollution"].iloc[:50],label="pollution")
plt.plot(seasonally_adjusted_data.iloc[:50],label="Seasonally Adjusted line")
plt.title("Seasonally adjusted data vs Original data for air pollution")
plt.xlabel('Date')
plt.ylabel('Pollution PM2.5 levels')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.show()

# detrended_data = S+R
# plt.figure(figsize=(12,9))
# plt.plot(df["pollution"].iloc[:50],label="pollution")
# plt.plot(detrended_data.iloc[:50],label="Detrended line")
# plt.title("Detrended data vs Original data for air pollution")
# plt.xlabel('Date')
# plt.ylabel('Pollution PM2.5 levels')
# plt.xticks(rotation=45)
# plt.grid()
# plt.legend()
# plt.show()

strength_of_trend = max(0,(1-(np.var(R)/(np.var(T+R)))))
print("The strength of trend for the air pollution levels is = "+ str(np.round(strength_of_trend*100,3))+"%")

strength_of_seasonality = max(0,(1-(np.var(R)/(np.var(S+R)))))
print("The strength of Seasonality for the air pollution levels is = "+ str(np.round((strength_of_seasonality)*100,3))+ "%")


################
# Holt-Winter Method
################
print("\n\nHolt Winters Method")
holtt = ets.ExponentialSmoothing(y_train,trend=None,damped=False,seasonal=None).fit()
holtf = holtt.forecast(steps=len(y_test))
holtf = pd.DataFrame(holtf).set_index(y_test.index)
MSE = np.square(np.subtract(y_test.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for simple exponential smoothing is ", MSE)


fig, ax = plt.subplots()
ax.plot(y_train,label= "Train Data")
ax.plot(y_test,label= "Test Data")
ax.plot(holtf,label= "Simple Exponential Smoothing")

plt.legend(loc='upper left')
plt.title('Simple Exponential Smoothing- Air Passengers')
plt.xlabel('Time (Hourly)')
plt.ylabel('Air pollution levels (PM2.5 concentration)')
plt.show()


y_train+=1
holtt = ets.ExponentialSmoothing(y_train,trend='multiplicative',damped=True,seasonal=None).fit()
holtf = holtt.forecast(steps=len(y_test))
holtf = pd.DataFrame(holtf).set_index(y_test.index)
MSE = np.square(np.subtract(y_test.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for double exponential smoothing is ", MSE)
fig, ax = plt.subplots()
ax.plot(y_train,label= "Train Data")
ax.plot(y_test,label= "Test Data")
ax.plot(holtf,label= "Holt's Linear Trend Method")
plt.legend(loc='upper left')
plt.title("Holt's Linear Trend Method")
plt.xlabel('Time (Hourly)')
plt.ylabel('Air pollution levels (PM2.5 concentration)')
plt.show()



holtt = ets.ExponentialSmoothing(y_train,trend='multiplicative',damped=True,seasonal='mul').fit()
holtf = holtt.forecast(steps=len(y_test))
holtf = pd.DataFrame(holtf).set_index(y_test.index)
MSE = np.square(np.subtract(y_test.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error for holt-winter method is ", MSE)


fig, ax = plt.subplots()
ax.plot(y_train,label= "Train data")
ax.plot(y_test,label= "Test data")
ax.plot(holtf,label= "Holt-Winter Method")
plt.legend(loc='upper left')
plt.title('Air pollution levels (PM2.5 concentration)')
plt.xlabel('Time (Hourly)')
plt.ylabel('Air Pollution levels')
plt.show()

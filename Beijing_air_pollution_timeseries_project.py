from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning

pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6313)
print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)
print("x_test shape: ",x_test.shape)
print("y_test shape: ",y_test.shape)


################
# Stationarity
################
print("\n\n\nStationarity Test:")
# Plot rolling mean and variance of Air pollution
Plot_Rolling_Mean_Var(df,"pollution")

# ADF test for original dataset
print("\nADF Test for Beijing Air Pollution dataset:")
ADF_Cal(df.loc[:,"pollution"])

# KPSS test for original dataset
print("\nKPSS Test for Beijing Air Pollution dataset:")
kpss_test(df.loc[:,"pollution"])

# Applying first order differencing
df["First Order Differencing"]=0
for i in range(1,len(df)):
    # First Order Differencing
    df["First Order Differencing"].iloc[i]= (
            df["pollution"].iloc[i] -
            df["pollution"].iloc[i-1])

df.to_csv("df_stationarized.csv")
df_stationarized = pd.read_csv("df_stationarized.csv")

# Plot rolling mean and variance of first order differenced dataset
Plot_Rolling_Mean_Var(df_stationarized, "First Order Differencing")

# ADF test
print("\nADF Test for First Order transformation:")
ADF_Cal(df.loc[:,"First Order Differencing"])

# KPSS test
print("\nKPSS Test for First Order transformation:")
kpss_test(df.loc[:,"First Order Differencing"])

Cal_autocorrelation(df["pollution"],20,"ACF function of Beijing Air pollution")
Cal_autocorrelation(df["First Order Differencing"],20,"ACF function of First Order Differencing of air pollution")

################
# Time Series Decomposition
################
print("\n\nTime Series Decomposition")

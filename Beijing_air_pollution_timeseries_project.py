import math
from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
import statsmodels.tsa.holtwinters as ets
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
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
print("Dataframe shape: "+str(df.shape))
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
# df = pd.get_dummies(df,columns=["wnd_dir"])

# Label Encoding
label_encoder = LabelEncoder()
df['wnd_dir_label'] = label_encoder.fit_transform(df['wnd_dir'])
df.drop(columns=["wnd_dir"],inplace=True)

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
df_stationarized = pd.read_csv("df_stationarized.csv",index_col="date")


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

# Preprocessing data for stationarized data (splitting into train and test)
df_stationarized.index = pd.to_datetime(df_stationarized.index)
df_stationarized.rename(columns={'First Order Differencing': 'pollution_new'}, inplace=True)
y_stationarized=df_stationarized["pollution_new"]
x_stationarized=df_stationarized.loc[:, df_stationarized.columns != 'pollution']
x_stationarized=x_stationarized.loc[:, x_stationarized.columns != 'pollution_new']
x_stationarized_train, x_stationarized_test, y_stationarized_train, y_stationarized_test = train_test_split(x_stationarized, y_stationarized, test_size=0.2, random_state=6313, shuffle= False)
print("y stationarized_test shape: ",y_stationarized_test.shape)
print("y stationarized_train shape: ",y_stationarized_train.shape)
print("x stationarized_train shape: ",x_stationarized_train.shape)

# Time Series plot of stationarized data
plt.figure()
df_stationarized["pollution_new"].plot(label="Stationarized Pollution")
plt.xlabel("Date")
plt.ylabel("Pollution Levels")
plt.title("Time Series plot of Air Pollution levels from stationarized dataset")
plt.grid()
plt.legend()
plt.show()

################
# Time Series Decomposition
################
print("\n\nTime Series Decomposition")
# takes a while to run for annual period. Keeping it as 24 for now
stl = STL(df["pollution"],period=24) # period = 365/8760, pls change later; tbd
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
Cal_autocorrelation(np.subtract(y_test.values,np.ndarray.flatten(holtf.values)),lags=20, title="ACF of residuals of Holt Winter method")
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



holtt = ets.ExponentialSmoothing(y_train,trend="multiplicative",damped= True,seasonal='mul').fit()
holtf = holtt.forecast(steps=len(y_test))
holtf = pd.DataFrame(holtf).set_index(y_test.index)
MSE = np.square(np.subtract(y_test.values,np.ndarray.flatten(holtf.values))).mean()
print("Mean square error of forecast errors for holt-winter method is ", MSE)
Q = sm.stats.acorr_ljungbox(np.subtract(y_test.values,np.ndarray.flatten(holtf.values)), lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q = ",Q)
print("Mean of forecast errors:",np.mean(np.subtract(y_test.values,np.ndarray.flatten(holtf.values))))
print("Variance of forecast errors:",np.var(np.subtract(y_test.values,np.ndarray.flatten(holtf.values))))

fig, ax = plt.subplots()
ax.plot(y_train,label= "Train data")
ax.plot(y_test,label= "Test data")
ax.plot(holtf,label= "Holt-Winter Method")
plt.legend(loc='upper left')
plt.title('Air pollution levels (PM2.5 concentration)')
plt.xlabel('Time (Hourly)')
plt.ylabel('Air Pollution levels')
plt.show()

y_train-=1
################
# Feature Selection
## Dimensionality Reduction
### backward stepwise regression, SVD, condition number, VIF analysis
################
# Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# transform data
# scaled_x_train = scaler.fit_transform(x_stationarized_train)
# scaled_x_train = pd.DataFrame(scaled_x_train, columns =x_stationarized.columns)
# print("Scaled x_trains shape:",scaled_x_train.shape)
# scaled_x_test = scaler.fit_transform(x_stationarized_test)
# scaled_x_test = pd.DataFrame(scaled_x_test, columns =x_stationarized.columns)
# print("Scaled x_test shape:",scaled_x_test.shape)
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_train = pd.DataFrame(scaled_x_train, columns =x_train.columns)
print("Scaled x_trains shape:",scaled_x_train.shape)
scaled_x_test = scaler.fit_transform(x_test)
scaled_x_test = pd.DataFrame(scaled_x_test, columns =x_test.columns)
print("Scaled x_test shape:",scaled_x_test.shape)


print("################")
print("Feature Selection")
print("################")
X1 = scaled_x_train.to_numpy()
H = np.dot(X1.T, X1)
u, s, vh = np.linalg.svd(H)
print('Singular values =', s)
print('Condition number is', round(np.linalg.cond(X1), 2))

print("\n\nOLS Function")
model = sm.OLS(y_train.values, sm.add_constant(scaled_x_train)).fit()
print(model.summary())

print("\n\nBackward Stepwise Regression")
flag=True
remaining_columns=scaled_x_train.columns
backward_stepwise_dataframe_results=pd.DataFrame(columns=["Columns/Features","AIC","BIC","Adjusted R-squared"])
while flag:
    model = sm.OLS(y_train.values, sm.add_constant(scaled_x_train[remaining_columns])).fit()
    print(model.summary())
    if(model.pvalues[model.pvalues.index != 'const'].sort_values(ascending=False)[0]>0.05):
        column_to_remove = (model.pvalues[model.pvalues.index != 'const'].sort_values(ascending=False).head(1).index[0])
        remaining_columns=remaining_columns.difference(column_to_remove.split())
        backward_stepwise_dataframe_results=pd.concat([backward_stepwise_dataframe_results,(pd.DataFrame([[remaining_columns,model.aic,model.bic, model.rsquared_adj]],
                                    columns=["Columns/Features","AIC","BIC","Adjusted R-squared"]))],axis=0,ignore_index=True)

    else:
        print("FINAL:")
        print("Remaining Columns: ", remaining_columns)
        print("Columns eliminated: ", scaled_x_train.columns.difference(remaining_columns))
        flag = False

print(backward_stepwise_dataframe_results)

print("\n\nVIF Test")
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data1 = pd.DataFrame()
vif_data1['feature'] = scaled_x_train.columns
vif_data1['VIF'] = [variance_inflation_factor(scaled_x_train.values, i) for i in range(len(scaled_x_train.columns))]
vif_data1=(vif_data1.sort_values(by=["VIF"],ascending=False))
print("VIF data:")
print(vif_data1)
columns_retained=scaled_x_train.columns
columns_to_be_eliminated=[]
vif_dataframe_results=pd.DataFrame(columns=["Columns/Features","AIC","BIC","Adjusted R-squared"])
flag=True
while flag:
    if (vif_data1["VIF"].iloc[0]>10): # play around with threshold value - 5 or 10
        column_to_be_removed=vif_data1["feature"].iloc[0]
        columns_to_be_eliminated.append(column_to_be_removed)
        vif_data1 = pd.DataFrame()
        columns_retained=columns_retained.difference(column_to_be_removed.split())
        vif_data1['feature'] = columns_retained
        vif_data1['VIF'] = [variance_inflation_factor(scaled_x_train[columns_retained].values, i) for i in
                            range(len(scaled_x_train[columns_retained].columns))]
        vif_data1 = (vif_data1.sort_values(by=["VIF"], ascending=False))

        # getting metrics
        model = sm.OLS(y_train.values, sm.add_constant(scaled_x_train[columns_retained])).fit()
        vif_dataframe_results = pd.concat([vif_dataframe_results, (
            pd.DataFrame([[columns_retained, model.aic, model.bic, model.rsquared_adj]],
                         columns=["Columns/Features", "AIC", "BIC", "Adjusted R-squared"]))], axis=0, ignore_index=True)
    else:
        flag=False
print("Columns retained: ", columns_retained)
print("Columns eliminated", columns_to_be_eliminated)
print(vif_dataframe_results)

print("Final OLS with columns: ",remaining_columns)
model = sm.OLS(y_train.values, sm.add_constant(scaled_x_train[columns_retained])).fit()
print(model.summary())

# Prediction
y_predicted = model.predict(sm.add_constant(scaled_x_test[columns_retained]))

prediction_error = y_test.values - y_predicted.values
y_predicted.index = y_test.index
Cal_autocorrelation(prediction_error,50,"ACF of prediction errors: Linear Regression")

# Prediction plot
plt.plot(y_train,label="Trained Pollution values")
plt.plot(y_test,label="Test Pollution values")
plt.plot(y_predicted,label="Predicted Pollution values")
plt.grid()
plt.title("Predicting pollution levels using Linear Regression model ")
plt.xlabel('Sample')
plt.ylabel('Pollution')
plt.legend()
plt.show()

################
# Linear Regression and Analysis
################
y_train_pred = model.predict(sm.add_constant(scaled_x_train[columns_retained]))
mse_test_linear_regression = np.mean(np.square(y_test.values - y_predicted.values))
print("MSE test value = ",mse_test_linear_regression)
mse_train_linear_regression = np.mean(np.square(y_train.values - y_train_pred.values))
print("MSE train value = ",mse_train_linear_regression)
rmse_test_linear_regression = np.sqrt(mse_test_linear_regression)
print("RMSE test value = ",rmse_test_linear_regression)
# Q-value calculation
print("Q-value:")
acf,shifted_indices, T = Cal_autocorrelation(prediction_error,50,title="ACF of residuals",show_fig=False)
Q = len(prediction_error)*np.sum(np.square(acf[50:]))
DOF = 50 - 8 #(number of lags being considered - number of parameters estimated in your model (including any intercept and coefficients))
alfa = 0.01
from scipy.stats import chi2
chi_critical = chi2.ppf(1-alfa, DOF)
print("Q-value = ",Q)
print("Chi Critical value = ", chi_critical)
if Q< chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")
lags = 50
print("Mean of residuals = ",np.mean(prediction_error))
print("Variance of residuals = ",np.var(prediction_error))


################
# Base Models
################
# Average
mse_average_training, mse_average_test, average_var_residual_error, average_var_forecase_error, q_test_value_average = average_forecast(y_train.values.tolist(),y_test.values.tolist())

# Naive
mse_naive_training, mse_naive_test, naive_var_residual_error, naive_var_forecase_error, q_test_value_naive = naive_base_model(y_train.values.tolist(),y_test.values.tolist())

# Drift
mse_drift_training, mse_drift_test, drift_var_residual_error, drift_var_forecase_error, q_test_value_drift = drift_base_model(y_train.values.tolist(),y_test.values.tolist())

# SES
mse_ses_training, mse_ses_test, ses_var_residual_error, ses_var_forecase_error, q_test_value_ses = ses_base_model(y_train.values.tolist(),y_test.values.tolist())




################
# GPAC table:
################
# non-seasonal order retrieval
ry = (sm.tsa.stattools.acf((y_train), nlags=200))
gpac=Cal_GPAC(ry,8,8)

# Parameter estimation:
## takes time to run
na=4
nb=0
model = sm.tsa.statespace.SARIMAX(y_stationarized_train,order=(na,0,nb),seasonal_order=(0,1,0,24)).fit()
print(model.summary())
# Prediction
# Predict test values
test_predictions = model.get_forecast(steps=len(y_stationarized_test))
model_hat = test_predictions.predicted_mean

# Predicting trained values:
# Predict training values
train_predictions = model.get_prediction(start=y_stationarized_train.index[0], end=y_stationarized_train.index[-1])
model_hat_train = train_predictions.predicted_mean
# model_hat_train = model.predict(start=y_stationarized_train.index[0],end=y_stationarized_train.index[-1])
#======================================
# Residuals Testing and Chi-Square test
sarima_forecast_error = y_stationarized_test-model_hat
sarima_residual_error = y_stationarized_train-model_hat_train
# ACF_PACF_Plot(sarima_residual_error,50)
Cal_autocorrelation(sarima_residual_error, 20, 'ACF of residuals')
acf,shifted_indices,T=Cal_autocorrelation(sarima_residual_error, 40, 'ACF of residuals',show_fig=False)
Cal_GPAC(acf[20:],7,7)
# Cal_autocorrelation(sarima_forecast_error, 100, 'ACF of forecast errors')
DOF = lags - na-nb
alfa = 0.01
from scipy.stats import chi2
chi_critical = chi2.ppf(1-alfa, DOF)
Q = sm.stats.acorr_ljungbox(sarima_residual_error, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q = ",sm.stats.acorr_ljungbox(sarima_residual_error, lags=[20], boxpierce=True, return_df=True)['bp_stat'].values[0])
print("Chi Critical = ",chi2.ppf(1-alfa, 20-na-nb))
from statsmodels.stats.diagnostic import acorr_ljungbox
bp_result = acorr_ljungbox(sarima_residual_error,lags=20,boxpierce=True)
if Q< chi_critical:
    print("The residual is white ")
else:
    print("The residual is NOT white ")
print(sm.stats.acorr_ljungbox(sarima_residual_error, lags=[lags]))
# plt.figure(figsize=(12,10))
# plt.plot(y_stationarized_test[:500],'r', label = "True data")
# plt.plot(model_hat[:500],'b', label = "Predicted data")
# plt.xticks(rotation=45)
# plt.xlabel("Samples")
# plt.ylabel("Magnitude")
# plt.legend()
# plt.title("ststsmodels SARIMA test parameter estimation and prediction")
# plt.show()


plt.figure(figsize=(12,10))
plt.plot(y_stationarized_train[:500],'r', label = "Train data")
plt.plot(model_hat_train[:500],'b', label = "Predicted data")
plt.xlabel("Samples")
plt.ylabel("Magnitude")
plt.xticks(rotation=45)
plt.legend()
plt.title("ststsmodels SARIMA trained values vs prediction")
plt.show()

print("Mean of residuals = ",np.mean(sarima_forecast_error))
print("Variance of residuals = ",np.var(sarima_forecast_error))
mse_test_sarima = np.mean(np.square(sarima_forecast_error))
print("MSE test value of SARIMA model = ",mse_test_sarima)
mse_train_sarima = np.mean(np.square(y_stationarized_train.values - model_hat_train.values))
print("MSE train value of SARIMA model = ",mse_train_sarima)
rmse_test_sarima = np.sqrt(mse_test_sarima)
print("RMSE test value of SARIMA model = ",rmse_test_sarima)

# LM algorithm
theta_new, sse_new, variance_hat, cov_mat,sse_list = lm_algorithm(np.array(y_stationarized_train.dropna()),4,0,len(y_stationarized_train))
print("theta = ",theta_new)
print("\nCovariance Matrix: \n",cov_mat)
print("Variance of error: ",variance_hat)
# sse_vs_iteration_plot(sse_list)

# inverse differencing of predicted values for plotting
predicted_values = abs(y_stationarized_test.shift(1)) + abs(model_hat)
# Prediction plot
plt.plot(y_train,label="Training Price")
plt.plot(y_test,label="Test Price")
plt.plot(predicted_values,label="Predicted Price")
plt.grid()
plt.title("Predicting prices using SARIMA model ")
plt.xlabel('Sample')
plt.ylabel('Pollution')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import statsmodels.api as sm
import seaborn as sns
import copy
import math
from scipy import signal

def Cal_rolling_mean_var(df,col):
    df_Mean_var = pd.DataFrame(columns=["Mean", "Variance"])
    # for first row:
    # df_Mean_var.loc[0, "Variance"] = 0
    # df_Mean_var.loc[0, ["Mean"]] = df[col].iloc[0]
    for i in range(1,len(df)):
        dfi = df[:i+1]
        df_Mean_var.at[i,"Mean"]=dfi.loc[:,col].mean()
        df_Mean_var.at[i, "Variance"] = dfi.loc[:, col].var()
    return df_Mean_var

def Plot_Rolling_Mean_Var(df,col):
    df_Mean_Var = Cal_rolling_mean_var(df,col)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot( df_Mean_Var.loc[:,"Mean"], label="Varying Mean")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Magnitude")
    ax1.set_title("Rolling Mean - "+col)
    ax1.legend()

    ax2.plot( df_Mean_Var.loc[:,"Variance"], label="Varying Variance")
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Rolling Variance - "+col)

    plt.tight_layout()
    plt.legend()
    plt.show()

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)

def Cal_autocorrelation(y,lags, title="ACF function",show_fig=True):
    y_mean = np.mean(y)
    T=len(y)
    acf_single_sided=list()
    for lag in range(lags+1):
        numerator=0
        denominator =0
        for i in range(lag,T):
            numerator += (y[i]-y_mean)*(y[i-lag]-y_mean)
        for i in range(T):
            denominator+=(y[i]-y_mean) **2
        correlation=numerator/denominator
        acf_single_sided.append(np.round(correlation,2))
    acf_reverse=acf_single_sided[::-1]
    acf = acf_reverse+acf_single_sided[1:]
    shifted_indices = [int(i - len(acf) // 2) for i in range(len(acf))]
    # print(shifted_indices)
    if show_fig != True:
        return acf,shifted_indices, T
    # plt.figure()
    (markers, stemlines, baseline) = plt.stem(shifted_indices,acf, markerfmt = 'o')
    plt.setp(markers, color = 'red')
    m = 1.96/np.sqrt(T)
    plt.axhspan(-m,m, alpha = 0.2, color = 'blue')
    plt.xlabel("Lags")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def average_forecast(yt, yt_forecast):
    # Average Forecasts
    print("\n\nAVERAGE FORECAST")
    # Creation of 1-step prediction
    y_hat = [0]
    residual_error_average = [0]
    sse = [0]
    for i in range(1, len(yt)):
        y_hat.append(np.round(np.sum(yt[:i]) / i, 3))
        residual_error_average.append(np.round(yt[i] - y_hat[i], 3))
        sse.append(np.round(residual_error_average[i] * residual_error_average[i], 3))
    time = np.arange(1, len(yt) + 1)
    average_forecast_1_step = pd.DataFrame(list(zip(time, yt, y_hat, residual_error_average, sse)),
                                           columns=["time", "yt", "y_hat", "error", "sum of squared error"])
    # print(average_forecast_1_step.to_string(index=False))

    # Creation of h-step prediction table
    y_hat_flat_prediction = [np.mean(yt)] * len(yt_forecast)
    y_hat_flat_prediction = [round(elem, 2) for elem in y_hat_flat_prediction]
    time_forecast = np.arange(1, len(yt_forecast) + 1)
    forecast_error = [a_i - b_i for a_i, b_i in zip(yt_forecast, y_hat_flat_prediction)]
    forecast_error = [round(elem, 2) for elem in forecast_error]
    forecast_sse = [i * i for i in forecast_error]
    forecast_sse = [round(elem, 2) for elem in forecast_sse]
    average_forecast_h_step = pd.DataFrame(
        list(zip(time_forecast, yt_forecast, y_hat_flat_prediction, forecast_error, forecast_sse)),
        columns=["h-step", "y(t+h)", "y_hat_forecast", "Forecast error", "sum of squared forecast error"])
    # print(average_forecast_h_step.to_string(index=False))

    # Plotting:
    average_forecast_h_step["y(t+h)"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    average_forecast_h_step["y_hat_forecast"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    plt.plot(time, yt, label="Training data")
    plt.plot(average_forecast_h_step["y(t+h)"], label="Test data")
    plt.plot(average_forecast_h_step["y_hat_forecast"], label="Average Method h-step prediction")
    plt.grid()
    plt.title("Average method & Forecast")
    plt.xlabel("Time")
    plt.ylabel("Pollution")
    plt.legend()
    plt.show()

    # MSE calculation
    mse_average_training = np.mean(sse[2:])
    mse_average_test = np.mean(forecast_sse)
    mse_average_results = {
        "Type of Error": ["Prediction Error (MSE)", "Forecast Error (MSE)"],
        "MSE Value": [np.round(mse_average_training, 3), np.round(mse_average_test, 3)]
    }

    df_average_mse = pd.DataFrame(mse_average_results)
    print(df_average_mse.to_string())

    # Variance calculation
    variance_prediction_average = np.var(residual_error_average[2:])
    print("Variance of prediction error: ", np.round(np.var(residual_error_average[2:]), 3))
    print("Variance of forecast error: ", np.round(np.var(forecast_error), 3))

    # Box-pierce test
    lags = 5
    acf, shifted_indices, T = Cal_autocorrelation(residual_error_average, 20, show_fig=False)
    acf_ = (acf[1:len(acf) // 2:])
    sum_squared_acf = np.sum([i ** 2 for i in acf_])
    number_of_observation = len(residual_error_average)
    q_test_value_average = number_of_observation * sum_squared_acf
    print("Q-test value = ", np.round(q_test_value_average, 3))

    # Also confirming q-test value from in-built function
    from statsmodels.stats.diagnostic import acorr_ljungbox
    bp_result = acorr_ljungbox(residual_error_average, lags=[20], boxpierce=True)
    print("Q-value from in-built function = ", np.round(bp_result["bp_stat"].iloc[0], 3))
    Cal_autocorrelation(forecast_error,lags=50,title="ACF function of Average Forecast errors")
    return mse_average_training, mse_average_test, np.round(np.var(residual_error_average[2:])), np.round(np.var(forecast_error), 3), q_test_value_average


def naive_base_model(yt,yt_forecast):
    # NAIVE METHOD
    print("\n\n NAIVE METHOD")
    # Creation of 1-step prediction
    y_hat = [0]
    residual_error_naive = [0]
    sse = [0]
    for i in range(1, len(yt)):
        y_hat.append(yt[i - 1])
        residual_error_naive.append(yt[i] - y_hat[i])
        sse.append(residual_error_naive[i] * residual_error_naive[i])
    time = np.arange(1, len(yt) + 1)
    naive_1_step = pd.DataFrame(list(zip(time, yt, y_hat, residual_error_naive, sse)),
                                columns=["time", "yt", "y_hat", "error", "sum of squared error"])
    # print(naive_1_step.to_string(index=False))

    # Creation of h-step prediction table
    y_hat_flat_prediction = [yt[-1]] * len(yt_forecast)
    time_forecast = np.arange(1, len(yt_forecast) + 1)
    forecast_error = [a_i - b_i for a_i, b_i in zip(yt_forecast, y_hat_flat_prediction)]
    forecast_sse = [i * i for i in forecast_error]
    naive_forecast_h_step = pd.DataFrame(
        list(zip(time_forecast, yt_forecast, y_hat_flat_prediction, forecast_error, forecast_sse)),
        columns=["h-step", "y(t+h)", "y_hat_forecast", "Forecast error", "sum of squared forecast error"])
    # print(naive_forecast_h_step.to_string(index=False))

    # Plotting:
    naive_forecast_h_step["y(t+h)"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    naive_forecast_h_step["y_hat_forecast"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    plt.plot(time, yt, label="Training data")
    plt.plot(naive_forecast_h_step["y(t+h)"], label="Test data")
    plt.plot(naive_forecast_h_step["y_hat_forecast"], label="Naive Method h-step prediction")
    plt.grid()
    plt.title("Naive method & Forecast")
    plt.xlabel("Time")
    plt.ylabel("Pollution")
    plt.legend()
    plt.show()

    # MSE calculation
    mse_naive_training = np.mean(sse[2:])
    mse_naive_test = np.mean(forecast_sse)
    mse_naive_results = {
        "Type of Error": ["Prediction Error (MSE)", "Forecast Error (MSE)"],
        "MSE Value": [np.round(mse_naive_training, 3), np.round(mse_naive_test, 3)]
    }

    df_naive_mse = pd.DataFrame(mse_naive_results)
    print(df_naive_mse.to_string())

    # Variance calculation
    variance_prediction_naive = np.var(residual_error_naive[2:])
    print("Variance of prediction error: ", np.round(np.var(residual_error_naive[2:]), 3))
    print("Variance of forecast error: ", np.round(np.var(forecast_error), 3))

    # Box test
    lags = 5
    acf, shifted_indices, T = Cal_autocorrelation(forecast_error, 5, show_fig=False)
    acf_ = (acf[1:len(acf) // 2:])
    sum_squared_acf = np.sum([i ** 2 for i in acf_])
    number_of_observation = len(forecast_error)
    q_test_value_naive = number_of_observation * sum_squared_acf
    print("Q-test value = ", np.round(q_test_value_naive, 3))

    # Also confirming q-test value from in-built function
    from statsmodels.stats.diagnostic import acorr_ljungbox
    bp_result = acorr_ljungbox(forecast_error, lags=[20], boxpierce=True)
    print("Q-value from in-built function = ", np.round(bp_result["bp_stat"].iloc[0], 3))
    Cal_autocorrelation(forecast_error,lags=50,title="ACF Function of forecast errors in Naive Base Model")
    return mse_naive_training, mse_naive_test, np.round(np.var(residual_error_naive[2:])), np.round(np.var(forecast_error), 3), q_test_value_naive

def drift_base_model(yt,yt_forecast):
    # Drift method
    print("\n\nDRIFT METHOD")
    # Creation of 1-step prediction
    y_hat = [0, 0]
    residual_error_drift = [0, 0]
    sse = [0, 0]
    for i in range(2, len(yt)):
        y_hat.append(np.round(yt[i - 1] + ((yt[i - 1] - yt[0]) / ((i - 1))), 3))
        residual_error_drift.append(np.round(yt[i] - y_hat[i], 3))
        sse.append(np.round(residual_error_drift[i] * residual_error_drift[i], 3))
    time = np.arange(1, len(yt) + 1)
    drift_1_step = pd.DataFrame(list(zip(time, yt, y_hat, residual_error_drift, sse)),
                                columns=["time", "yt", "y_hat", "error", "sum of squared error"])
    # print(drift_1_step.to_string(index=False))

    # Creation of h-step prediction table
    y_hat_flat_prediction = []
    for i in range(1, len(yt_forecast) + 1):
        y_hat_flat_prediction.append(yt[-1] + (i * ((yt[-1] - yt[0]) / (len(yt) - 1))))
    time_forecast = np.arange(1, len(yt_forecast) + 1)
    forecast_error = [a_i - b_i for a_i, b_i in zip(yt_forecast, y_hat_flat_prediction)]
    forecast_sse = [i * i for i in forecast_error]
    forecast_sse = [round(elem, 2) for elem in forecast_sse]
    drift_forecast_h_step = pd.DataFrame(
        list(zip(time_forecast, yt_forecast, y_hat_flat_prediction, forecast_error, forecast_sse)),
        columns=["h-step", "y(t+h)", "y_hat_forecast", "Forecast error", "sum of squared forecast error"])
    # print(drift_forecast_h_step.to_string(index=False))

    # Plotting:
    drift_forecast_h_step["y(t+h)"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    drift_forecast_h_step["y_hat_forecast"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    plt.plot(time, yt, label="Training data")
    plt.plot(drift_forecast_h_step["y(t+h)"], label="Test data")
    plt.plot(drift_forecast_h_step["y_hat_forecast"], label="Drift Method h-step prediction")
    plt.grid()
    plt.title("Drift method & Forecast")
    plt.xlabel("Time")
    plt.ylabel("Pollution")
    plt.legend()
    plt.show()

    # MSE calculation
    mse_drift_training = np.mean(sse[2:])
    mse_drift_test = np.mean(forecast_sse)
    mse_drift_results = {
        "Type of Error": ["Prediction Error (MSE)", "Forecast Error (MSE)"],
        "MSE Value": [np.round(mse_drift_training, 3), np.round(mse_drift_test, 3)]
    }

    df_drift_mse = pd.DataFrame(mse_drift_results)
    print(df_drift_mse.to_string())

    # Variance calculation
    variance_prediction_drift = np.var(residual_error_drift[2:])
    print("Variance of prediction error: ", np.round(np.var(residual_error_drift[2:]), 3))
    print("Variance of forecast error: ", np.round(np.var(forecast_error), 3))

    # Box test
    lags = 5
    acf, shifted_indices, T = Cal_autocorrelation(forecast_error, 5, show_fig=False)
    acf_ = (acf[1:len(acf) // 2:])
    sum_squared_acf = np.sum([i ** 2 for i in acf_])
    number_of_observation = len(forecast_error)
    q_test_value_drift = number_of_observation * sum_squared_acf
    print("Q-test value = ", np.round(q_test_value_drift, 3))

    # Also confirming q-test value from in-built function
    from statsmodels.stats.diagnostic import acorr_ljungbox
    bp_result = acorr_ljungbox(forecast_error, lags=[20], boxpierce=True)
    print("Q-value from in-built function = ", np.round(bp_result["bp_stat"].iloc[0], 3))
    Cal_autocorrelation(forecast_error, lags=50,title="ACF Function of forecast errors in Drift Method")
    return mse_drift_training, mse_drift_test, np.round(np.var(residual_error_drift[2:])), np.round(
        np.var(forecast_error), 3), q_test_value_drift

def ses_base_model(yt,yt_forecast):
    # SES method
    print("\n\nSES Method")
    # Creation of 1-step prediction
    y_hat = [0]
    y_hat.append(np.round(yt[0], 3))
    residual_error_ses = [0]
    residual_error_ses.append(np.round(yt[1] - y_hat[1], 3))
    sse = [0]
    sse.append(residual_error_ses[1] * residual_error_ses[1])
    alfa = 0.5
    for i in range(2, len(yt)):
        y_hat.append(np.round(((alfa) * yt[i - 1]) + ((1 - alfa) * y_hat[i - 1]), 3))
        residual_error_ses.append(np.round(yt[i] - y_hat[i], 3))
        sse.append(np.round(residual_error_ses[i] * residual_error_ses[i], 3))
    time = np.arange(1, len(yt) + 1)
    ses_1_step = pd.DataFrame(list(zip(time, yt, y_hat, residual_error_ses, sse)),
                              columns=["time", "yt", "y_hat", "error", "sum of squared error"])
    # print(ses_1_step.to_string(index=False))

    # Creation of h-step prediction table
    y_hat_flat_prediction = []
    y_hat_flat_prediction = [((alfa * yt[-1]) + ((1 - alfa) * (y_hat[-1])))] * len(yt_forecast)
    y_hat_flat_prediction = [round(elem, 2) for elem in y_hat_flat_prediction]
    time_forecast = np.arange(1, len(yt_forecast) + 1)
    forecast_error = [a_i - b_i for a_i, b_i in zip(yt_forecast, y_hat_flat_prediction)]
    forecast_error = [round(elem, 2) for elem in forecast_error]
    forecast_sse = [i * i for i in forecast_error]
    forecast_sse = [round(elem, 2) for elem in forecast_sse]
    ses_forecast_h_step = pd.DataFrame(
        list(zip(time_forecast, yt_forecast, y_hat_flat_prediction, forecast_error, forecast_sse)),
        columns=["h-step", "y(t+h)", "y_hat_forecast", "Forecast error", "sum of squared forecast error"])
    # print(ses_forecast_h_step.to_string(index=False))

    # Plotting:
    ses_forecast_h_step["y(t+h)"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    ses_forecast_h_step["y_hat_forecast"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    plt.plot(time, yt, label="Training data")
    plt.plot(ses_forecast_h_step["y(t+h)"], label="Test data")
    plt.plot(ses_forecast_h_step["y_hat_forecast"], label="SES Method h-step prediction")
    plt.grid()
    plt.title("SES method & Forecast")
    plt.xlabel("Time")
    plt.ylabel("Pollution")
    plt.legend()
    plt.show()

    # MSE calculation
    mse_ses_training = np.mean(sse[2:])
    mse_ses_test = np.mean(forecast_sse)
    mse_ses_results = {
        "Type of Error": ["Prediction Error (MSE)", "Forecast Error (MSE)"],
        "MSE Value": [np.round(mse_ses_training, 3), np.round(mse_ses_test, 3)]
    }

    df_ses_mse = pd.DataFrame(mse_ses_results)
    print(df_ses_mse.to_string())

    # Variance calculation
    variance_prediction_ses = np.var(residual_error_ses[2:])
    print("Variance of prediction error: ", np.round(np.var(residual_error_ses[2:]), 3))
    print("Variance of forecast error: ", np.round(np.var(forecast_error), 3))

    # Box test
    lags = 5
    acf, shifted_indices, T = Cal_autocorrelation(forecast_error, 5, show_fig=False)
    acf_ = (acf[1:len(acf) // 2:])
    sum_squared_acf = np.sum([i ** 2 for i in acf_])
    number_of_observation = len(forecast_error)
    q_test_value_ses = number_of_observation * sum_squared_acf
    print("Q-test value = ", np.round(q_test_value_ses, 3))

    # Also confirming q-test value from in-built function
    from statsmodels.stats.diagnostic import acorr_ljungbox
    bp_result = acorr_ljungbox(forecast_error, lags=[20], boxpierce=True)
    print("Q-value from in-built function = ", np.round(bp_result["bp_stat"].iloc[0], 3))
    Cal_autocorrelation(forecast_error,lags=50,title="ACF function of forecast errors using SES method")

    alfa1 = 0
    alfa2 = 0.25
    alfa3 = 0.75
    alfa4 = 0.99
    y_hat_flat_prediction_alfa1 = [((alfa1 * yt[-1]) + ((1 - alfa1) * (y_hat[-1])))] * len(yt_forecast)
    y_hat_flat_prediction_alfa2 = [((alfa2 * yt[-1]) + ((1 - alfa2) * (y_hat[-1])))] * len(yt_forecast)
    y_hat_flat_prediction_alfa3 = [((alfa3 * yt[-1]) + ((1 - alfa3) * (y_hat[-1])))] * len(yt_forecast)
    y_hat_flat_prediction_alfa4 = [((alfa4 * yt[-1]) + ((1 - alfa4) * (y_hat[-1])))] * len(yt_forecast)
    time_forecast = np.arange(1, len(yt_forecast) + 1)
    alfa_comparison_df = pd.DataFrame(list(
        zip(time, yt_forecast, y_hat_flat_prediction_alfa1, y_hat_flat_prediction_alfa2, y_hat_flat_prediction_alfa3,
            y_hat_flat_prediction_alfa4)),
                                      columns=["h-step", "y(t+h)", "y_hat_forecast_alfa1", "y_hat_forecast_alfa2",
                                               "y_hat_forecast_alfa3", "y_hat_forecast_alfa4"])

    # alfa_comparison_df["y_hat_forecast_alfa1"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    # alfa_comparison_df["y_hat_forecast_alfa2"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    # alfa_comparison_df["y_hat_forecast_alfa3"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    # alfa_comparison_df["y_hat_forecast_alfa4"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    # alfa_comparison_df["y(t+h)"].index = np.arange(len(yt) + 1, len(yt) + len(yt_forecast) + 1)
    # num_rows = 2
    # num_cols = 2
    # fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    # axs = axs.ravel()
    # fig.suptitle("SES Forecasting with various alfa values")
    # axs[0].plot(time, yt, label="Training")
    # axs[0].plot(alfa_comparison_df["y(t+h)"], label="Test")
    # axs[0].plot(alfa_comparison_df["y_hat_forecast_alfa1"], label="alfa=0")
    # axs[0].set_title(f'Alfa =0')
    # axs[0].set_xlabel('Time')
    # axs[0].set_ylabel('Pollution')
    # axs[0].grid()
    # axs[0].legend()
    #
    # axs[1].plot(time, yt, label="Training")
    # axs[1].plot(alfa_comparison_df["y(t+h)"], label="Test")
    # axs[1].plot(alfa_comparison_df["y_hat_forecast_alfa2"], label="alfa=0.25")
    # axs[1].set_title(f'Alfa =0.25')
    # axs[1].set_xlabel('Time')
    # axs[1].set_ylabel('Pollution')
    # axs[1].grid()
    # axs[1].legend()
    #
    # axs[2].plot(time, yt, label="Training")
    # axs[2].plot(alfa_comparison_df["y(t+h)"], label="Test")
    # axs[2].plot(alfa_comparison_df["y_hat_forecast_alfa3"], label="alfa=0.75")
    # axs[2].set_title(f'Alfa =0.75')
    # axs[2].set_xlabel('Time')
    # axs[2].set_ylabel('Pollution')
    # axs[2].grid()
    # axs[2].legend()
    #
    # axs[3].plot(time, yt, label="Training")
    # axs[3].plot(alfa_comparison_df["y(t+h)"], label="Test")
    # axs[3].plot(alfa_comparison_df["y_hat_forecast_alfa4"], label="alfa=0.99")
    # axs[3].set_title(f'Alfa =0.99')
    # axs[3].set_xlabel('Time')
    # axs[3].set_ylabel('Pollution')
    # axs[3].grid()
    # axs[3].legend()
    # plt.show()
    return mse_ses_training, mse_ses_test, np.round(np.var(residual_error_ses[2:])), np.round(
        np.var(forecast_error), 3), q_test_value_ses



def ACF_PACF_Plot(y,lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

def Cal_GPAC(Ry,k_index=7,j_index=7):
    indices = list(range(-(len(Ry)//2),(len(Ry)//2)+1))
    index_mapping = dict(zip(indices, Ry))
    print(index_mapping)
    phi = np.zeros((j_index,k_index-1))

    for k in range(1,k_index):
        for j in range(j_index):
            if k==1:
                numerator = index_mapping[j+k]
                denominator = index_mapping[j]
                if denominator==0:
                    final_value = np.inf
                else:
                    final_value=numerator/denominator
                phi[j][k-1]=np.round(final_value,3)
            else:
                denominator=np.array(np.zeros((k,k)),dtype=np.float64)
                numerator = np.array(np.zeros((k, k)),dtype=np.float64)
                numerator[:, -1] = [index_mapping[i] for i in range(j+1, j+k+1)]
                ji=copy.deepcopy(j)
                for ki in range(k):
                    denominator[:,-1-ki]=[index_mapping[i] for i in range(ji-k+1, ji+1)]
                    ji+=1
                numerator[:, :-1] = denominator[:, :-1]

                # if np.linalg.det(numerator) ==0:  # ==0:
                #     final_value = 0
                if np.linalg.det(denominator) ==0:#<0.00000001:
                    final_value=np.inf
                else:
                    final_value=np.linalg.det(numerator)/np.linalg.det(denominator)
                phi[j][k - 1] = np.round(final_value,3)


    plt.figure(figsize=(40, 16))
    sns.heatmap(phi, annot=True, fmt='.3f',robust=True)
    new_column_labels = list(range(1,k_index))
    plt.xticks(np.arange(len(new_column_labels)) + 0.5, new_column_labels)
    plt.title("GPAC Table",fontsize = 20)
    plt.show()
    return phi



def differencing(y, d, s):
    diff=copy.deepcopy(y)
    # print("diff before while loop: ",diff)
    if s!=0:
        while(d>=0):
            for i in range(len(diff)-1,s-1,-1):
                diff[i] = diff[i] - diff[i-s]
            # print("diff after 1st for loop: ",diff)
            for j in range(s):
                # print(diff)
                # print("j=",j)
                diff[j]=np.nan
            d-=1
    if s==0:
        while (d > 0):
            for i in range(len(diff) - 1, d - 1, -1):
                diff[i] = diff[i] - diff[i - d]
            # print("diff after 1st for loop: ", diff)
            for j in range(d):
                diff[j] = np.nan
            d -= 1
    # print(diff)
    cleanedList = [x for x in diff if not math.isnan(x)]
    return cleanedList



# LM algorithm confidence interval
def lm_algorithm_confidence_interval(na,nb,theta_new,cov_mat):
    lower_bound=[]
    upper_bound=[]
    for i in range(na+nb):
        if (cov_mat[i][i] <0):
            lower_bound.append(theta_new[i] + 2 * (-1*np.sqrt(abs(cov_mat[i][i]))))
            upper_bound.append(theta_new[i] - 2 * (-1*np.sqrt(abs(cov_mat[i][i]))))
        else:
            lower_bound.append(theta_new[i] - 2*np.sqrt(cov_mat[i][i]))
            upper_bound.append(theta_new[i] + 2 * np.sqrt(cov_mat[i][i]))
    # print("lower bound : ",lower_bound)
    # print("upper bound : ",upper_bound)
    return lower_bound,upper_bound


# finding roots of num and den
def roots_zero_pole_check(na,nb,theta_new):
    num = theta_new[na:]
    den = theta_new[:na]
    num_roots = np.roots(np.insert(num,0,1))
    den_roots = np.roots(np.insert(den,0,1))
    print("Roots of numerator = ",np.round(num_roots,3))
    print("Roots of Denominator = ",np.round(den_roots,3))
    return num_roots, den_roots


# Plot of SSE vs # iterations
def sse_vs_iteration_plot(sse_list):
    plt.plot(sse_list)
    plt.xticks(list(range(len(sse_list))),list(range(1,len(sse_list)+1)))
    plt.xlabel("Number of Iterations")
    plt.ylabel("Sum of Squared Errors")
    plt.title("Sum of Squared Errors vs Number of Iterations")
    plt.show()

def Calculate_e_for_lm(theta,na,nb,y):
    num = theta[na:]
    den = theta[:na]
    len_diff = len(num)-len(den)
    if (len_diff>0):
        den=np.append(den,[0]*len_diff)
    else:
        len_diff = len(den) - len(num)
        num = np.append(num,[0] * len_diff)
    num = np.insert(num, 0, 1)
    den = np.insert(den, 0, 1)
    system = (den, num, 1)
    t, e = signal.dlsim(system, y)
    return e


def lm_algorithm(y,na,nb,T):
    MAX_iterations = 100
    delta = 0.000001
    epsilon = 0.1
    mu_max=100**20
    mu=0.01
    n = na+nb
    variance_hat=0
    cov_mat = 0
    sse_list=[]

    # step 0
    theta=np.zeros(na+nb).reshape(na+nb,1)
    iterations = 0
    e = np.array(y.copy()).reshape(len(y), 1)
    for iterations in range(MAX_iterations):
        # step 1
        e = Calculate_e_for_lm(theta, na, nb, y)
        sse = np.dot(np.transpose(e), (e))
        sse_list.append(sse.item())
        X = np.zeros(shape=(T, na + nb))
        for i in range(na + nb):
            thetai = copy.deepcopy(theta)
            thetai[i] += delta
            ei = Calculate_e_for_lm(thetai, na, nb, y)
            xi = (e - ei) / delta
            thetai[i] -= delta
            X[:, i] = xi[:, 0]
        A = np.dot(np.transpose(X), (X))
        g = np.dot(np.transpose(X), (e))

        # step 2
        identity = np.identity(n)
        delta_theta = np.matmul(np.linalg.inv(A + (mu * identity)), g)
        theta_new = theta + delta_theta
        e_new = Calculate_e_for_lm(theta_new, na, nb, y)
        sse_new = np.dot(np.transpose(e_new), (e_new))

        # step 3
        if iterations < MAX_iterations:
            if sse_new <= sse:
                if (np.linalg.norm(delta_theta) < epsilon):
                    theta_hat=theta_new
                    variance_hat = sse_new/(T-n)
                    cov_mat = variance_hat*np.linalg.inv(A)
                    print("Algorithm has converged!!!")
                    # print(f"The Estimated Parameters are: ",theta_hat)
                    # print(f"The Covariance matrix is: ",cov_mat)
                    # print(f"The Variance of error is: ",variance_hat)
                    return theta_new, sse_new, variance_hat, cov_mat, sse_list
                else:
                    theta = theta_new
                    mu=mu/10
            while sse_new >=sse:
                mu=mu*10
                if mu > mu_max:
                    print("mu greater than mu_max. Algorithm has not converged.")
                    return
                # return to step 2
                identity = np.identity(n)
                delta_theta = np.matmul(np.linalg.inv(A + (mu * identity)), g)
                theta_new = theta + delta_theta
                e_new = Calculate_e_for_lm(theta_new, na, nb, y)
                sse_new = np.dot(np.transpose(e_new), (e_new))


            #iterations+=1
        if (iterations > MAX_iterations):
            print("Reached the limit of maximum number of iterations. Algorithm has not converged.")
            return
        theta = theta_new
    return theta_new, sse_new, variance_hat, cov_mat, sse_list



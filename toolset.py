import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

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

##################################
# Triple Exponential Smoothing (TES)
# HOLT-WINTERS
##################################

# TES = SES (Level, a) + DES (Trend, B) + Mevsimsellik (Seasonality)

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

warnings.filterwarnings("ignore")

data = sm.datasets.co2.load_pandas()

y = data.data

y = y["co2"].resample("MS").mean()  # weekly to monthly

y.isnull().sum()  # 5 NaN

y.fillna(y.bfill(), inplace=True)

y.plot(color="pink")
plt.grid()
plt.xlabel("Year")
plt.ylabel("CO2")
plt.show()


################
# HOLDOUT
################

train = y[: "1997-12-12"]
len(train)

test = y["1998-01-01":]
len(test)

################################
# Zaman Serisi Yapısal Analizi
################################

# Durağanlık Testi - Stationary Test (Dickey-Fuller Test)

def is_stationary(y):

    # H0: non-stationary
    # H1: stationary

    p_value = sm.tsa.stattools.adfuller(y)[1]

    if p_value < 0.05:
        print("HO Rejected and time series is stationary, p-value: {}".format(round(p_value, 4)))
        print("tr\tSonuç:  Durağan, p-value: {}".format(round(p_value, 3)))
    else:
        print("HO not Rejected and time series is not stationary, p-value: {}".format(round(p_value, 4)))
        print("tr\tSonuç:  Durağan değil, p-value: {}".format(round(p_value, 3)))

is_stationary(y)

def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, "k", label="Original " + model)

    axes[1].plot(result.trend, label="Trend", color="gray")
    axes[1].legend(loc="upper left")

    axes[2].plot(result.seasonal, "purple", label="Seasonality & Mean: " + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc="upper left")

    axes[3].plot(result.resid, "pink", label="Residuals & Mean: " + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc="upper left")

    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompose(y, stationary=True)

def compare_train_test_pred(train, test, prediction, title):
    mae = mean_absolute_error(test, prediction)
    train.plot(legend=True, label="TRAIN", color="blue", title=title + " MAE: " + str(round(mae, 3)))
    test.plot(legend=True, label="TEST", color="red")
    prediction.plot(legend=True, label="PREDICTION", color="green")
    plt.grid()
    plt.xlabel("Year")
    plt.ylabel("CO2")
    plt.show()

# Model
tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit(smoothing_level=0.5,  # a
                                                                                              smoothing_trend=0.5,  # b
                                                                                              smoothing_seasonal=0.5) #g

y_pred = tes_model.forecast(len(test))

mean_absolute_error(test, y_pred)
# add is better than mul!, then we chose add

compare_train_test_pred(train, test, y_pred, "Triple Exponential Smoothing (Holt-Winters) ")


#############################
# Hyperparameter Optimization
#############################

def tes_optimzer(train, alphas, betas, gamas, step=len(test)):
    best_alpha, best_beta, best_gama, best_mae = None, None, None, float("inf")
    for a in alphas:
        for b in betas:
            for g in gamas:
                tes_model = (ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).
                             fit(smoothing_level=a, smoothing_trend=b, smoothing_seasonal=g))

                y_pred = tes_model.forecast(step)
                mae = mean_absolute_error(test, y_pred)
                if mae < best_mae:
                    best_alpha, best_beta, best_gama, best_mae = a, b, g, mae
                print("alpha: {}\tbeta: {}\tgama: {}\tmae: {}\t".format(round(a, 3),
                                                                        round(b, 3),
                                                                        round(g, 3),
                                                                        round(mae, 4)))

    print("best_alpha: {}\nbest_beta: {}\nbest_gama: {}\nbest_mae: {}".format(round(best_alpha, 3),
                                                                              round(best_beta, 3),
                                                                              round(best_gama, 3),
                                                                              round(best_mae, 4)))

    return round(best_alpha, 3), round(best_beta, 3), round(best_gama, 3), round(best_mae, 3)

alphas = np.arange(0.2, 1, 0.1)
betas = np.arange(0.2, 1, 0.1)
gamas = np.arange(0.2, 1, 0.1)

best_alpha, best_beta, best_gama, best_mae = tes_optimzer(train, alphas, betas, gamas)

##############
# 2. way:
##############
"""alphas = betas = gammas = np.arange(0.2, 1, 0.1)

abg = list(itertools.product(alphas, betas, gammas))

def tes_optimzer2(train, abg, step=len(test)):

    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")

    for i in abg:
            tes_model = (ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).
                        fit(smoothing_level=i[0], smoothing_trend=i[1], smoothing_seasonal=i[2]))

            y_pred = tes_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)

            if mae < best_mae:
                best_alpha, best_beta, best_gamma, best_mae = i[0], i[1], i[2], mae
            print("alpha: {}\tbeta: {}\tgama: {}\tmae: {}\t".format(round(i[0], 3),
                                                                    round(i[1], 3),
                                                                    round(i[2], 3),
                                                                    round(mae, 4)))

    print("best_alpha: {}\nbest_beta: {}\nbest_gama: {}\nbest_mae: {}".format(round(best_alpha, 3),
                                                                              round(best_beta, 3),
                                                                              round(best_gamma, 3),
                                                                              round(best_mae, 4)))

    return round(best_alpha, 3), round(best_beta, 3), round(best_gamma, 3), round(best_mae, 3)


best_alpha, best_beta, best_gamma, best_mae = tes_optimzer2(train, abg)"""

# Final Model

final_tes_model = (ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).
                   fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gama))

y_pred = final_tes_model.forecast(len(test))
print("Best Parameters-->", final_tes_model.params)

compare_train_test_pred(train, test, y_pred, "Triple Exponential Smoothing (Hold-Winters)")

# Forecasting

final_tes_model = (ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12).
                   fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gama))

final_tes_model.forecast(3)
final_tes_model.forecast(int(input("How many months, do you want to forecast?")))













##################################
# Double Exponential Smoothing (DES)
##################################

# DES: Level (SES) + Trend
# y(t) = Level + Trend + Seasonality + Noise (add)
# y(t) = Level * Trend * Seasonality * Noise (mul)

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

warnings.filterwarnings("ignore")

data = sm.datasets.co2.load_pandas()

y = data.data

y = y["co2"].resample("MS").mean()  # haftalıktan aylığa çevirdik (weekly to monthly)

y.isnull().sum()  # 5 NaN

y.fillna(y.bfill(), inplace=True)  # boş gözlem, bir sonraki değer ile dolduruldu.

y.plot()
plt.grid()
plt.xlabel("Year")
plt.ylabel("CO2")
plt.show()

###############
# Holdout
###############

train = y[: "1997-12-01"]
len(train)  # 478 ay

test = y["1998-01-01":]
len(test)  # 48


################################
# Zaman Serisi Yapısal Analizi
################################

# Durağanlık Testi - Stationary Test (Dickey-Fuller Test)

def is_stationary(y):

    # H0: Non-Stationary
    # H1: Stationary

    p_value = sm.tsa.stattools.adfuller(y)[1]

    if p_value < 0.05:
        print("en\nResult: Stationary (H0: non-stationary), p-value: {}".format(round(p_value, 3)))
        print("tr\nSonuç:  Durağan, p-value: {}".format(round(p_value, 3)))
    else:
        print("en\nResult: Non-Stationary (H0: non-stationary), p-value: {}".format(round(p_value, 3)))
        print("tr\nSonuç:  Durağan değil, p-value: {}".format(round(p_value, 3)))

is_stationary(y)

# Zaman Serisi Bileşenleri ve Durağanlık Testi

def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, "k", label="Original " + model)

    axes[1].plot(result.trend, label="Trend")
    axes[1].legend(loc="upper left")

    axes[2].plot(result.seasonal, "g", label="Seasonality & Mean: " + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc="upper left")

    axes[3].plot(result.resid, "r", label="Residuals & Mean: " + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc="upper left")

    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompose(y, stationary=True)

# Model
des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5, smoothing_trend=0.5)

y_pred = des_model.forecast(48)

mean_absolute_error(test, y_pred)
def compare_train_test_pred(train, test, prediction, title):
    mae = mean_absolute_error(test, prediction)

    train["1985":].plot(legend=True, label="TRAIN", title=title + " MAE: " + str(round(mae, 4)), color="blue")
    test.plot(legend=True, label="TEST", color="purple")
    prediction.plot(legend=True, label="PREDICTION", color="orange")
    plt.grid()
    plt.xlabel("Years")
    plt.ylabel("CO2")
    plt.show()

compare_train_test_pred(train, test, y_pred, "Double Exponential Smoothing")

#############################
# Hyperparameter Optimization
#############################

def des_optimizer(train, alphas, betas, step=len(test)):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for i in alphas:
        for j in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=i,
                                                                     smoothing_slope=j)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)

            if mae < best_mae:
                best_alpha, best_beta, best_mae = i, j, mae
            print("alpha:", round(i, 2), "beta:", round(j, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.1)
betas = np.arange(0.01, 1, 0.1)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas)

######################
# Final DES Model
######################
# y(t) = Level + Trend + Seasonality + Noise
final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha,
                                                               smoothing_slope=best_beta)

y_pred = final_des_model.forecast(48)
compare_train_test_pred(train, test, y_pred, "Double Exponential Smoothing")

# y(t) = Level * Trend * Seasonality * Noise
final_des_model = ExponentialSmoothing(train, trend="mul").fit(smoothing_level=best_alpha,
                                                               smoothing_trend=best_beta)

y_pred = final_des_model.forecast(48)
compare_train_test_pred(train, test, y_pred, "Double Exponential Smoothing")
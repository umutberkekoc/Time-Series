######################################
# Smoothing Methods (Holt-Winters)
######################################
# Single Exponential Smoothing

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
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

test = y["1998-01-01": ]
len(test)  # 48


################################
# Zaman Serisi Yapısal Analizi
################################

# Durağanlık Testi (Dickey-Fuller Test)

def is_stationary(y):

    # H0: Non-Stationary
    # H1: Stationary

    p_value = sm.tsa.stattools.adfuller(y)[1]

    if p_value < 0.05:
        print("Result: Stationary (H0: non-stationary, p-value: {}".format(round(p_value, 3)))
    else:
        print("Result: Non-Stationary (H0: non-stationary, p-value: {}".format(round(p_value, 3)))

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


#################################
# Single Exponential Smoothing
#################################

# SES = Level (trend ve mevsimsellik (seasonality) olmamalı)

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)
# smoothing level = alpha = 0.5

y_pred = ses_model.forecast(48)  # test seti eleman sayısı giriliyor

mean_absolute_error(test, y_pred)

train.plot(title="Single Exponential Smoothing", color="blue")
test.plot(color="red")
y_pred.plot(color="green")
plt.xlabel("Year")
plt.ylabel("C02")
plt.show()

train["1983":].plot(title="Single Exponential Smoothing", color="blue")
test.plot(color="red")
y_pred.plot(color="green")
plt.xlabel("Year")
plt.ylabel("C02")
plt.grid()
plt.show()

def compare_train_test_pred(train, test, prediction, title):
    mae = mean_absolute_error(test, prediction)

    train["1985":].plot(legend=True, label="TRAIN", title=title + " MAE: " + str(round(mae, 3)), color="blue")
    test.plot(legend=True, label="TEST", color="red")
    prediction.plot(legend="True", label="PREDICTION", color="green")
    plt.grid()
    plt.xlabel("Years")
    plt.ylabel("CO2")
    plt.show()

compare_train_test_pred(train, test, y_pred, "Single Exponential Smoothing")

print("Parameters: ", ses_model.params)  # alpha 0.5 girdiğimiz için 0.5 geldi

#############################
# Hyperparameter Optimization
#############################

def ses_optimizer(train, alphas, step=len(test)):
    best_alpha, best_mae = None, float("inf")

    for i in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=i)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_mae = i, mae
        print("Alpha:", round(i, 3), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 3), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.6, 1, 0.01)
best_alpha, best_mae = ses_optimizer(train, alphas)


ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(48)
mean_absolute_error(test, y_pred)

compare_train_test_pred(train, test, y_pred, "Single Exponential Smoothing")

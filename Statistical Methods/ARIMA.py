# ARIMA (p, d, q): Autoregressive Integrated Moving Average

import itertools
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

data = sm.datasets.co2.load_pandas()
y = data.data
y = y["co2"].resample("MS").mean()
y = y.fillna(y.bfill())

train = y[: "1997-12-01"]
len(train)
test = y["1998-01-01":]
len(test)

def plot_co2(train, test, prediction, title):

    mae = mean_absolute_error(test, prediction)

    train["1985":].plot(legend=True, label="TRAIN", title=title + ", MAE: " + str(round(mae, 3)))
    test.plot(legend=True, label="TEST", color="red")
    prediction.plot(legend=True, label="PREDICTION", color="green")
    plt.grid()
    plt.xlabel("Year")
    plt.ylabel("co2")
    plt.show()


model = sm.tsa.arima.ARIMA(train, order=(1, 1, 1))
arima_model = model.fit()

arima_model.summary()

y_pred = arima_model.forecast(len(test))
y_pred = pd.Series(y_pred, index=test.index)
mean_absolute_error(test, y_pred)

plot_co2(train, test, y_pred, title="ARIMA")

############################################################
# Hyperparameter Optimization (Model Derecelerini Belirleme)
############################################################
def arima_optimizer(train, orders):
    best_params, best_aic = None, float("inf")

    for i in orders:
        try:
            arima_model = sm.tsa.arima.ARIMA(train, order=i).fit()
            aic = arima_model.aic

            if aic < best_aic:
                best_params, best_aic = i, aic
            print("ARIMA: {}\tAIC: {}".format(i, round(aic, 2)))
        except:
            continue
    print("Best ARIMA: {}\t Best AIC: {}".format(best_params, round(best_aic, 2)))
    return best_params

p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))

best_params_aic = arima_optimizer(train, pdq)


#################
# Final Model
#################

final_arima_model = sm.tsa.arima.ARIMA(train, order=best_params_aic).fit()

y_pred = final_arima_model.forecast(len(test))
y_pred = pd.Series(y_pred, index=test.index)
mean_absolute_error(test, y_pred)

plot_co2(train, test, y_pred, title="ARIMA With\nHyperparameter Optimization")




















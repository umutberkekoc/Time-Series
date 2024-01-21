# SARIMA (p, d, q): Seasonal Autoregressive Integrated Moving Average

import itertools
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
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
    train["1985": ].plot(legend=True, label="TRAIN", title=title + ", MAE: " + str(round(mae, 3)))
    test.plot(legend=True, label="TRAIN", color="red")
    prediction.plot(legend=True, label="PREDICTION", color="green")
    plt.grid()
    plt.xlabel("year")
    plt.ylabel("co2")
    plt.show()


model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12))
sarima_model = model.fit(disp=0)

y_pred_test = sarima_model.get_forecast(steps=len(test))

y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "SARIMA")


#############################
# Hyperparameter Optimization
#############################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(i[0], i[1], i[2], 12) for i in list(itertools.product(p, d, q))]


def sarima_optimizer(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), None, None

    for i in pdq:
        for j in seasonal_pdq:
            try:
                sarima_model = SARIMAX(train, order=i, seasonal_order=j)
                results = sarima_model.fit(disp=0)
                aic = results.aic

                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, i, j
                print("SARIMA: {}X{}12 - aıc: {}".format(i, j, aic))
            except:
                continue
    print("SARIMA: {}X{}12 - aıc: {}".format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order, best_aic

best_order, best_seasonal_order, best_aic = sarima_optimizer(train, pdq, seasonal_pdq)


##############
# Final Model
##############

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
final_sarima_model = model.fit(disp=0)

y_pred_test = final_sarima_model.get_forecast(steps=len(test))

y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "SARIMA")


##########################################
# BONUS: MAE'ye Göre SARIMA Optimizasyonu
##########################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(i[0], i[1], i[2], 12) for i in list(itertools.product(p, d, q))]

def sarima_optimizer(train, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), None, None

    for i in pdq:
        for j in seasonal_pdq:
            try:
                sarima_model = SARIMAX(train, order=i, seasonal_order=j)
                sarima_model = sarima_model.fit(disp=0)
                y_pred_test = sarima_model.get_forecast(steps=len(test))
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test, y_pred)

                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, i, j
                print("SARIMA: {}X{}12 - MAE: {}".format(i, j, mae))
            except:
                continue

    print("SARIMA: {}X{}12 - MAE: {}".format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order, best_mae

best_order, best_seasonal_order, best_mae = sarima_optimizer(train, pdq, seasonal_pdq)

##############
# Final Model
##############

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
final_sarima_model = model.fit(disp=0)

y_pred_test = final_sarima_model.get_forecast(steps=len(test))

y_pred = y_pred_test.predicted_mean
y_pred = pd.Series(y_pred, index=test.index)

plot_co2(train, test, y_pred, "SARIMA")


# Final Model 2:

model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal_order)
final_sarima_model = model.fit(disp=0)

feature_predict = final_sarima_model.get_forecast(steps=3)
feature_predict = feature_predict.predicted_mean

import math
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas_datareader.data as web

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

PLOT_LINE_WIDTH = 1.7
models = {
        'linear_regression': LinearRegression(n_jobs=-1),
        'lasso': Lasso(),
        'quadratic_regression_2': make_pipeline(PolynomialFeatures(2), Ridge()),
        'quadratic_regression_3': make_pipeline(PolynomialFeatures(3), Ridge())
    }


def prepare_data():
    # Read data from the web
    # start = datetime.datetime(2010, 1, 1)
    # end = datetime.datetime(2017, 1, 11)
    # df = web.DataReader("AAPL", 'yahoo', start, end)
    df = pd.read_csv('data/AAPL.csv', index_col='Date', parse_dates=True)

    dfreg = df.loc[:, ['Adj Close', 'Volume']]
    dfreg['HL_CT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)

    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.05 * len(dfreg)))

    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)

    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluaiton
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y_lately = y[-forecast_out:]
    y = y[:-forecast_out]

    return dfreg, X, y, X_lately, y_lately


def train_models(models_dict, X_train, y_train):
    for model_name, model in models_dict.items():
        model.fit(X_train, y_train)


def get_test_scores(models_dict, X_test, y_test):
    test_scores = dict()

    for model_name, model in models_dict.items():
        test_scores[model_name] = model.score(X_test, y_test)

    return test_scores


def plot_prediction_result(dfreg, forecast_set, last_date, method=''):
    dfreg['Forecast'+method] = np.nan

    last_unix = last_date

    next_unix = last_unix + datetime.timedelta(days=1)

    for i in forecast_set:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]

    dfreg['Forecast'+method].tail(300).plot(linewidth=PLOT_LINE_WIDTH)


def make_predictions(models_dict, X_lately):
    forecast_sets = dict()

    for model_name, model in models_dict.items():
        forecast_sets[model_name] = model.predict(X_lately)

    return forecast_sets


def plot_results(forecast_sets, dfreg):
    last_date = dfreg.iloc[-1].name

    for model_name, forecast_set in forecast_sets.items():
        plot_prediction_result(dfreg, forecast_set, last_date, method='_'+model_name)


def main():
    dfreg, X, y, X_lately, y_lately = prepare_data()

    # Separation of training and testing of model by cross validation train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_models(models, X_train, y_train)

    model_test_scores = get_test_scores(models, X_test, y_test)
    print(f'Models test scores: {model_test_scores}')

    # Retraining models on whole dataset
    train_models(models, X, y)

    # Making predictions
    forecast_sets = make_predictions(models, X_lately)

    # Plot results
    plot_results(forecast_sets, dfreg)

    dfreg['Adj Close'].tail(300).plot(linewidth=PLOT_LINE_WIDTH)
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.savefig('stock_forecast_result.png')
    plt.show()


if __name__ == '__main__':
    main()

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


def make_prediction():
    pass


def plot_prediction_result(dfreg, forecast_set, last_date, method=''):
    dfreg['Forecast'+method] = np.nan

    last_unix = last_date

    next_unix = last_unix + datetime.timedelta(days=1)

    for i in forecast_set:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]

    dfreg['Forecast'+method].tail(300).plot(linewidth=0.7)


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2017, 1, 11)

# Read data from the web
# df = web.DataReader("AAPL", 'yahoo', start, end)
df = pd.read_csv('data/AAPL.csv', index_col='Date', parse_dates=True)
print(df.tail())

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

# Separation of training and testing of model by cross validation train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Lasso regression
lasso = Lasso()
lasso.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test, y_test)
confidencepoly3 = clfpoly3.score(X_test, y_test)
confidencelasso = lasso.score(X_test, y_test)
confidence = lasso.score(X_test, y_test)

# Printing the forecast
forecast_set = lasso.predict(X_lately)
dfreg['Forecast'] = np.nan
print(forecast_set, confidence, forecast_out)

last_date = dfreg.iloc[-1].name

last_unix = last_date

next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]


dfreg['Adj Close'].tail(300).plot(linewidth=0.7)
dfreg['Forecast'].tail(300).plot(linewidth=0.7)
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


from datetime import date, timedelta
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.linear_model import LinearRegression
from train_model import trainModel, getFeatures, standardScaler


class DataAnlysis:
    def __init__(self, stock_name):
        self.df = yf.Ticker(stock_name).history(period="25mo").reset_index()

    def create_dtFeatures(self):
        """
        Input:
        Output:
            dataframe(pandas DataFrame): dataframe with the date time features
        """
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['day'] = self.df['Date'].dt.day
        self.df['month'] = self.df['Date'].dt.month
        self.df['year'] = self.df['Date'].dt.year
        return self.df

    def create_laggedFeatures(self, feature, shift_value):
        """
        Input:
            feature(string): the dataframe column
            shift_value(int): the value of the lagged feature
        Output:
            dataframe(pandas DataFrame): dataframe with the lagged feature
        """
        self.df[feature + '_lagged'] = self.df[feature].shift(shift_value)
        return self.df

    def detrend(self, feature):
        """
        Input:
            feature(string): feature that wil be detrended
        Output:
            detrend(np array): detrended values of features
            plots
        """
        X = [i for i in range(0, len(self.df))]
        X = np.reshape(X, (len(X), 1))
        y = self.df[feature].values
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        plt.plot(X, trend)
        plt.plot(X, self.df[feature].values)
        plt.title('{} and linear trend'.format(feature))
        plt.show()
        detrend = y - trend
        plt.plot(X, detrend)
        plt.title('{} detrend'.format(feature))
        plt.show()
        return detrend

    def ma_smoothing(self, k):
        """
        Input:
            k(int): range of moving average
        Output:
            dataframe(pandas DataFrame): dataframe with the ma value
        """
        index = 0
        k_ = int((k-1)/2)
        ma_smoothing = []
        while index < len(self.df)-k_:
            if index < k_:
                ma_smoothing.append(np.nan)
            else:    
                ma_smoothing.append(self.df[index-k_:index+k_+1]['Close'].mean())
            index += 1
        for _ in range(k_):
            ma_smoothing.append(np.nan)
        self.df['ma_'+str(k)] = ma_smoothing
        return self.df

class Modelling:
    def __init__(self, df):
        self.df = df
        self.TEST_SIZE = int(len(self.df)*(9/10))

    def calculate_RMSE(self, y_pred, y_true):
        """
        Input:
            y_pred(np vector): predicted values
            y_true(np vector): true values
        Output:
            RMSE(float): value of Root Mean Square Error
        """
        RMSE = (np.mean((y_pred - y_true)**2))**0.5
        return RMSE


    # Auto Regressive Model
    def AutoRegressive(self):
        avg_range = 7
        index = len(self.df) - 1
        X, y = [], []
        while index-avg_range+1 > 0:
            X.append(self.df['Close'][index-avg_range:index].values)
            y.append(np.array([self.df.iloc[index]['Close']]))
            index = index - 1
        X.reverse(), y.reverse()
        X, y = np.array(X), np.array(y)
        X_train, y_train = X[:self.TEST_SIZE], y[:self.TEST_SIZE]
        X_test, y_test = X[self.TEST_SIZE:], y[self.TEST_SIZE:]
        model_ar = LinearRegression()
        model_ar.fit(X_train, y_train)
        y_pred = model_ar.predict(X_test)
        y_true = y_test
        dates = self.df['Date'][self.TEST_SIZE:][7:]
        plt.plot(dates, y_pred, label='pred')
        plt.plot(dates, y_true, label='true')
        plt.xticks(rotation=45)
        plt.title('Auto Regressive model')
        plt.ylabel('BTC-value')
        plt.legend()
        plt.show()
        rmse_ar = self.calculate_RMSE(y_pred, y_true)
        print('RMSE of Auto Regressive: {}'.format(round(rmse_ar, 2)))
        return model_ar


    # Moving Average Model
    def MovingAverage(self):
        model_ar = self.AutoRegressive()
        avg_range = 7
        index = len(self.df) - 1
        X, y = [], []
        while index-avg_range+1 > 0:
            X.append(self.df['Close'][index-avg_range:index].values)
            y.append(np.array([self.df.iloc[index]['Close']]))
            index = index - 1
        X.reverse(), y.reverse()
        X, y = np.array(X), np.array(y)
        error = model_ar.predict(X) - y
        X_train, y_train = error[:self.TEST_SIZE], y[:self.TEST_SIZE]
        X_test, y_test = error[self.TEST_SIZE:], y[self.TEST_SIZE:]
        model_ma = LinearRegression()
        model_ma.fit(X_train, y_train)
        y_pred = model_ma.predict(X_test)
        c = np.mean(y_train)/3 # c constant
        y_pred = y_pred + c
        y_true = y_test
        dates = self.df['Date'][self.TEST_SIZE:][7:]
        plt.plot(dates, y_pred, label='pred')
        plt.plot(dates, y_true, label='true')
        plt.xticks(rotation=45)
        plt.title('Moving Average model')
        plt.ylabel('BTC-value')
        plt.legend()
        plt.show()
        rmse_ma = self.calculate_RMSE(y_pred, y_true)
        print('RMSE of Auto Regressive: {}'.format(round(rmse_ma, 2)))
        return model_ma

    # LSTM Model
    def LSTM_model(self):
        self.df['Date'] = pd.to_datetime(self.df['Date']).apply(lambda x: x.date())
        df_train = self.df[:self.TEST_SIZE]
        df_test = self.df[self.TEST_SIZE:]
        model_lstm = trainModel(df_train)
        index = 0
        y_pred = []
        dates = []
        while index+8 < len(df_test):
            df_test_ = df_test[index:index+8]
            vec = getFeatures(df_test_, len(df_test_))[:-1]
            z = standardScaler(vec)
            x = np.array([z[0]])
            x = x.reshape(x.shape[0], 1, x.shape[1])
            yPred = model_lstm.predict(x)
            minValue, maxValue = z[1][0], z[1][1]
            yPred = yPred*(maxValue - minValue) + minValue
            y_pred.append(yPred[0][0])
            dates.append(df_test_['Date'][-1:].values)
            index += 1
        y_true = df_test[7:len(df_test)-1]['Close'].values
        plt.plot(dates, y_pred, label='pred')
        plt.plot(dates, y_true, label='true')
        plt.xticks(rotation=45)
        plt.title('LSTM model')
        plt.ylabel('BTC-value')
        plt.legend()
        plt.show()
        rmse_lstm = self.calculate_RMSE(y_pred, y_true)
        print('RMSE of LSTM: {}'.format(round(rmse_lstm, 2)))
        return y_pred


# References
# Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. OTexts.com/fpp2. Accessed on <07/07/2020>.

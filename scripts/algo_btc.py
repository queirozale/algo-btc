from datetime import date, timedelta
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


class LoadData:
    def __init__(self, stock_name, month_range=36):
        print("Reading the data...")
        try:
            self.df = yf.Ticker(stock_name).history(period="{}mo".format(month_range)).reset_index()
            print("Successful data reading!")
        except:
            self.df = pd.DataFrame()
            print("Data reading error. Check if you typed the correct ticker code")

    def create_dtFeatures(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['day'] = self.df['Date'].dt.day
        self.df['month'] = self.df['Date'].dt.month
        self.df['year'] = self.df['Date'].dt.year
        print("DateTime features created!")

    def create_laggedFeatures(self, feature, shift_value):
        self.df[feature + '_lagged'] = self.df[feature].shift(shift_value)
        print("Lagged feature created!")

    def create_features(self):
        self.create_dtFeatures()
        self.create_laggedFeatures('Close', 1)
        print("Features created!")


class DataAnlysis:
    def __init__(self, df):
        self.df = df

    def detrend(self, feature):
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


class Preprocessing:
    def __init__(self, df, dev_rate):
        self.dev_index = int(len(df)*(1-dev_rate))
        self.df = df

    def get_minmax(self):
        def minmax_Scaler(x):
            xMin, xMax = np.min(x), np.max(x)
            results = {'scaled_data': [(k - xMin)/(xMax - xMin) for k in x],
                        'minmax': [xMin, xMax]}
            return results
        z = minmax_Scaler(self.df['Close'])
        self.df['Close_'+'minmax'] = z['scaled_data']
        return z['minmax']

    def sampling(self, avg_range=7):
        index = len(self.df) - 1
        X, y = [], []
        while index-avg_range+1 > 0:
            X.append(self.df['Close_minmax'][index-avg_range:index].values)
            y.append(np.array([self.df.iloc[index]['Close_minmax']]))
            index = index - 1
        X.reverse(), y.reverse()
        X, y = np.array(X), np.array(y)
        vectors = {'X_train': X[:self.dev_index],
                    'y_train': y[:self.dev_index],
                    'X_test': X[self.dev_index:],
                    'y_test': y[self.dev_index:],
                    'X': X, 
                    'y': y}
        return vectors


class Modelling:
    def __init__(self, vectors, minmax, step):
        if step == 'validation':
            self.X_train, self.y_train = vectors['X_train'], vectors['y_train']
            self.X_test, self.y_test = vectors['X_test'], vectors['y_test']
        elif step == 'test':
            self.X_train, self.y_train = vectors['X_train'], vectors['y_train']
        else:
            print("You did not input a valid step")
        self.minmax = minmax
        self.step = step

    def LinearReg(self):
        print("==========Starting Linear Regression...==========")
        lr = LinearRegression().fit(self.X_train, self.y_train)
        return lr

    def RandomForestReg(self):
        print("==========Starting Random Forest Regression...==========")
        # rfr GridSearch
        max_depth = [5, 10, 20]
        min_samples_split = [2, 6, 12]
        min_samples_leaf = [1, 4, 12]
        bootstrap = [True, False]
        param_dist = {'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        rfr = RandomForestRegressor(n_estimators=50, n_jobs=-1)
        print("Starting Random Forest Regressor Grid Search...")
        rfr_cv = RandomizedSearchCV(rfr, param_dist, cv=6, n_jobs=-1,
                                verbose=2, n_iter=400, scoring='neg_mean_absolute_error')
        rfr_cv.fit(self.X_train, self.y_train)
        return rfr_cv

    def MLP(self):
        print("==========Starting Multi-layer Perceptron...==========")
        mlp = Sequential()
        mlp.add(Dense(1, input_dim=7, activation='tanh'))
        mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        mlp.fit(self.X_train, self.y_train, epochs=20, batch_size=10)
        return mlp

    def LSTM_net(self):
        print("==========Starting LSTM Network...==========")
        NEURONS, BATCH_SIZE, NB_EPOCH = 4, 1, 20
        X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        model = Sequential()
        model.add(LSTM(NEURONS, batch_input_shape=(BATCH_SIZE, X_train.shape[1], 
                                                X_train.shape[2]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
        for i in range(NB_EPOCH):
            model.fit(X_train, self.y_train, epochs=1, batch_size=BATCH_SIZE, verbose=0, 
                    shuffle=False)
            model.reset_states()
            print("Epoch {} completed!".format(i+1))
        return model

    def predict_val(self):
        if self.step == 'test':
            print("step = test, not validation")
        elif self.step == 'validation':
            lr = self.LinearReg()
            rfr_cv = self.RandomForestReg()
            mlp = self.MLP()
            lstm_net = self.LSTM_net()

            def get_OriginalScale(values, minmax):
                valueMin, valueMax = minmax[0], minmax[1]
                vec_min = np.full(values.shape, valueMin)
                vec_max = np.full(values.shape, valueMax)
                return values*(vec_max-vec_min) + vec_min

            def get_pred(model):
                y_pred = get_OriginalScale(model.predict(self.X_test), self.minmax)
                print("Pred finished!")
                return y_pred

            def calculate_RMSE(y_pred, y_true):
                RMSE = (np.mean((y_pred - y_true)**2))**0.5
                return RMSE

            all_pred = {'LinearReg': get_pred(lr),
                        'RandomForestReg': get_pred(rfr_cv),
                        'MovingAverageSmoothing': get_OriginalScale(np.mean(self.X_test, axis=1).reshape(-1, 1), self.minmax),
                        'MLP': get_pred(mlp),
                        'LSTM_net': get_OriginalScale(lstm_net.predict(self.X_test.reshape(self.X_test.shape[0], 
                        1, self.X_test.shape[1])), self.minmax),
                        'y_true': get_OriginalScale(self.y_test, self.minmax)}
            for key in all_pred.keys():
                print("RMSE of {}: {}".format(key, round(calculate_RMSE(all_pred[key], all_pred['y_true']), 2)))
            return all_pred
        else:
            print("You did not input a valid step")

# References
# Hyndman, R.J., & Athanasopoulos, G. (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. OTexts.com/fpp2. Accessed on <07/07/2020>.

from algo_btc import LoadData, Preprocessing, Modelling
import matplotlib.pyplot as plt
import pickle
import os
from tensorflow.keras.models import load_model
import numpy as np

def main(action):
    STOCK_NAME = "BTC-USD"
    MODEL_PATH = "models"
    ld = LoadData(STOCK_NAME)
    ld.create_features()
    pp = Preprocessing(ld.df.copy(), 0.1)
    minmax = pp.get_minmax()
    vectors = pp.sampling()

    def run_validation():
        m = Modelling(vectors, minmax, 'validation')
        all_pred = m.predict_val()
        dates = ld.df['Date'][int(len(ld.df)*0.9)+7:]
        fig = plt.figure(figsize=[16, 5])
        for key in all_pred.keys():
            plt.plot(dates, all_pred[key], label=key)
        plt.legend()
        plt.title('Validation sample predictions')
        plt.savefig('val_pred.png')
        plt.show()

    def train_model():
        m = Modelling(vectors, minmax, 'test')
        # Creating the models
        lr = m.LinearReg()
        rfr_cv = m.RandomForestReg()
        mlp = m.MLP()
        lstm_net = m.LSTM_net()
        # Saving the models
        with open(os.path.join(MODEL_PATH, 'lr.pkl'), 'wb') as f:
            pickle.dump(lr, f)
        with open(os.path.join(MODEL_PATH, 'rfr_cv.pkl'), 'wb') as f:
            pickle.dump(rfr_cv, f)
        mlp.save(os.path.join(MODEL_PATH, 'mlp.h5'))
        lstm_net.save(os.path.joing(MODEL_PATH, 'lstm_net.h5'))
    
    def get_signals():
        # Loading models
        lr = pickle.load(open(os.path.join(MODEL_PATH, 'lr.pkl'), 'rb'))
        rfr_cv = pickle.load(open(os.path.join(MODEL_PATH, 'rfr_cv.pkl'), 'rb'))
        mlp = load_model(os.path.join(MODEL_PATH, 'mlp.h5'))
        lstm_net = load_model(os.path.join(MODEL_PATH, 'lstm_net.h5'))
        df = ld.df.copy()
        
        # Functions to process data
        def get_minmax():
            def minmax_Scaler(x):
                xMin, xMax = np.min(x), np.max(x)
                results = {'scaled_data': [(k - xMin)/(xMax - xMin) for k in x],
                            'minmax': [xMin, xMax]}
                return results
            z = minmax_Scaler(df['Close'])
            df['Close_'+'minmax'] = z['scaled_data']
            return z['minmax']

        def get_OriginalScale(values, minmax):
            valueMin, valueMax = minmax[0], minmax[1]
            vec_min = np.full(values.shape, valueMin)
            vec_max = np.full(values.shape, valueMax)
            return values*(vec_max-vec_min) + vec_min

        minmax = get_minmax()
        x = df['Close_minmax'][-8:-1].values.reshape(-1, 1).T

        # Function to create the signal
        def signal(pred, last_value):
            if pred > last_value:
                return 'hot'
            elif pred == last_value:
                return 'neutral'
            else:
                return 'cold'

        # Creating the preds
        preds = [lr.predict(x), rfr_cv.predict(x),
        mlp.predict(x), lstm_net.predict(x.reshape(x.shape[0], 
        1, x.shape[1])), np.mean(x, axis=1).reshape(-1, 1)]

        all_pred = {'LinearReg': get_OriginalScale(lr.predict(x), minmax),
                    'RandomForestReg': get_OriginalScale(rfr_cv.predict(x), minmax),
                    'MovingAverageSmoothing': get_OriginalScale(np.mean(x, axis=1).reshape(-1, 1), minmax),
                    'MLP': get_OriginalScale(mlp.predict(x), minmax),
                    'LSTM_net': get_OriginalScale(lstm_net.predict(x.reshape(x.shape[0], 
                                1, x.shape[1])), minmax)}

        print("=====TABLE OF SIGNALS AND PREDICTIONS=====")
        print("Date: {}".format(df['Date'][-1:].values))
        last_value = get_OriginalScale(x, minmax)[0][-1:][0]
        print("Last value: {}".format(round(last_value, 2)))
        for key in all_pred.keys():
            pred = all_pred[key]
            if len(pred.shape) == 2:
                pred = pred[0][0]
            else:
                pred = pred[0]
            sig = signal(pred, last_value)
            print("Model: {} --- Prediction: {} --- Signal: {}".format(key, round(pred, 2), sig))

    if action == 'validation':
        run_validation()
    elif action == 'train_model':
        train_model()
    elif action == 'get_signals':
        get_signals()
    else:
        print("Action not found")

if __name__ == "__main__":
    main('get_signals')
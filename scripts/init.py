from algo_btc import LoadData, Preprocessing, Modelling
import matplotlib.pyplot as plt
import pickle

def main():
    STOCK_NAME = "BTC-USD"
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
        with open('models/lr.pkl', 'wb') as f:
            pickle.dump(lr, f)
        with open('models/rfr_cv.pkl', 'wb') as f:
            pickle.dump(rfr_cv, f)
        mlp.save('models/mlp.h5')
        lstm_net.save('models/lstm_net.h5')
    
    train_model()

if __name__ == "__main__":
    main()
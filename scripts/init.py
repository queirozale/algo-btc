from algo_btc import LoadData, Preprocessing, Modelling
import matplotlib.pyplot as plt

def main():
    STOCK_NAME = "BTC-USD"
    ld = LoadData(STOCK_NAME)
    ld.create_features()
    pp = Preprocessing(ld.df.copy(), 0.1)
    minmax = pp.get_minmax()
    vectors = pp.sampling()
    X_train, y_train = vectors['X_train'], vectors['y_train']
    X_test, y_test = vectors['X_test'], vectors['y_test']
    m = Modelling(vectors, minmax)
    all_pred = m.predict()
    dates = ld.df['Date'][int(len(ld.df)*0.9)+7:]
    fig = plt.figure(figsize=[16, 5])
    for key in all_pred.keys():
        plt.plot(dates, all_pred[key], label=key)
    plt.legend()
    plt.title('Validation sample predictions')
    plt.savefig('val_pred.png')
    plt.show()


if __name__ == "__main__":
    main()
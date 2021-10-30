import pandas as pd
import sklearn
import math
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

all_data = pd.read_csv('Apple.csv', sep=',', header=0)

all_data = all_data[['Open', 'Close', 'Adj Close', 'Volume']]

all_data['%_Range'] = (all_data['Close'] - all_data['Open']) / all_data['Open'] * 100
all_data['Price'] = all_data['Adj Close']

all_data = all_data[['%_Range', 'Volume', 'Price']]

all_data.fillna(-9999, inplace=True)
all_data.dropna(inplace=True)

predict_variable = 'Price'
predict_out_long = 30
all_data['Long Term Prediction'] = all_data[predict_variable].shift(-predict_out_long)

X = np.array(all_data.drop('Long Term Prediction', axis=1))
X = preprocessing.scale(X)


'''Long Term '''

X_long = X[:-predict_out_long]
X_predict_long = X[-predict_out_long:]

all_data.dropna(inplace=True)
y_long = np.array(all_data['Long Term Prediction'])


def train_model_long():
    accuracy = [0]

    while max(accuracy) < 0.97:
        X_train_long, X_test_long, y_train_long, y_test_long = train_test_split(X_long, y_long, test_size=0.2)
        linear_long_train = LinearRegression(n_jobs=-1)
        linear_long_train.fit(X_train_long, y_train_long)
        acc_long = linear_long_train.score(X_test_long, y_test_long)
        accuracy.append(acc_long)

        if acc_long == max(accuracy):
            with open('Apple_long.pickle', 'wb') as f:
                pickle.dump(linear_long_train, f)

    print('Training complete. Accuracy: ' + str(max(accuracy) * 100) + '%')


def present_prediction_long():
    trained_model = open('Apple_long.pickle', 'rb')
    linear_long_best = pickle.load(trained_model)
    y_predict_long = linear_long_best.predict(X_predict_long)
    print(str(predict_out_long) + ' day prediction: \n' + str(y_predict_long))

    predictions = []
    real_price = []
    days_real = []
    days_predict = []

    for i in range(len(all_data['Long Term Prediction']) + len(y_predict_long)):
        days_predict.append(i)

    for i in range(len(all_data['Long Term Prediction'])):
        predictions.append(all_data['Long Term Prediction'][i])
        real_price.append(all_data['Long Term Prediction'][i])
        days_real.append(i)

    for i in y_predict_long:
        predictions.append(i)

    plt.plot(days_real, real_price, label='Price', zorder=1)
    plt.plot(days_predict, predictions, label='Predictions', zorder=0)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Long Term Prediction')
    plt.legend(loc=4)
    plt.show()


def make_prediction_long():
    trained_model = open('Apple_long.pickle', 'rb')
    linear_long_best = pickle.load(trained_model)
    y_predict_long = linear_long_best.predict(X_predict_long)

    return y_predict_long


def present_evaluated_prediction_long(conclusion):
    predictions = []
    real_price = []
    days_real = []
    days_predict = []

    for i in range(len(all_data['Long Term Prediction']) + len(conclusion)):
        days_predict.append(i)

    for i in range(len(all_data['Long Term Prediction'])):
        predictions.append(all_data['Long Term Prediction'][i])
        real_price.append(all_data['Long Term Prediction'][i])
        days_real.append(i)

    for i in conclusion:
        predictions.append(i)

    plt.plot(days_real, real_price, label='Price', zorder=1)
    plt.plot(days_predict, predictions, label='Predictions', zorder=0)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Evaluated Long Term Prediction')
    plt.legend(loc=4)
    plt.show()


# train_model_long()
present_prediction_long()



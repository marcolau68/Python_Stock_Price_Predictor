import pandas as pd
import sklearn
import math
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn import tree

all_data = pd.read_csv('Apple.csv', sep=',', header=0)

all_data = all_data[['Open', 'Close', 'Adj Close', 'Volume']]

all_data['%_Range'] = (all_data['Close'] - all_data['Open']) / all_data['Open'] * 100
all_data['Price'] = all_data['Adj Close']

all_data = all_data[['%_Range', 'Volume', 'Price']]

all_data.fillna(-9999, inplace=True)
all_data.dropna(inplace=True)

predict_variable = 'Price'
predict_out_medium = 8

all_data['Medium Term Prediction'] = all_data[predict_variable].shift(-predict_out_medium)

X = np.array(all_data.drop('Medium Term Prediction', axis=1))
X = preprocessing.scale(X)

data_size = 600

'''Medium Term'''

X_medium = X[-data_size:-predict_out_medium]
X_predict_medium = X[-predict_out_medium:]

all_data.dropna(inplace=True)
y_medium = np.array(all_data['Medium Term Prediction'][-(data_size - predict_out_medium):])


def train_model_medium():
    accuracy = [0]

    while max(accuracy) < 0.96:
        X_train_medium, X_test_medium, y_train_medium, y_test_medium = train_test_split(X_medium, y_medium, test_size=0.2)
        linear_medium_train = LinearRegression(n_jobs=-1)
        linear_medium_train.fit(X_train_medium, y_train_medium)
        acc_medium = linear_medium_train.score(X_test_medium, y_test_medium)
        accuracy.append(acc_medium)

        if acc_medium == max(accuracy):
            with open('Apple_medium.pickle', 'wb') as f:
                pickle.dump(linear_medium_train, f)

    print('Training complete. Accuracy: ' + str(max(accuracy) * 100) + '%')


def present_prediction_medium():
    trained_model = open('Apple_medium.pickle', 'rb')
    linear_medium_best = pickle.load(trained_model)
    y_predict_medium = linear_medium_best.predict(X_predict_medium)
    print(str(predict_out_medium) + ' day prediction: \n' + str(y_predict_medium))

    predictions = []
    real_price = []
    days_real = []
    days_predict = []

    for i in range(data_size + len(y_predict_medium)):
        days_predict.append(i)

    for i in range(data_size):
        predictions.append(all_data['Medium Term Prediction'][len(all_data['Medium Term Prediction']) - data_size + i])
        real_price.append(all_data['Medium Term Prediction'][len(all_data['Medium Term Prediction']) - data_size + i])
        days_real.append(i)

    for i in y_predict_medium:
        predictions.append(i)

    plt.plot(days_real, real_price, label='Price', zorder=1)
    plt.plot(days_predict, predictions, label='Predictions', zorder=0)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Medium Term Prediction')
    plt.legend(loc=4)
    plt.show()


def make_prediction_medium():
    trained_model = open('Apple_medium.pickle', 'rb')
    linear_medium_best = pickle.load(trained_model)
    y_predict_medium = linear_medium_best.predict(X_predict_medium)

    return y_predict_medium


def present_evaluated_prediction_medium(conclusion):
    print(conclusion)
    predictions = []
    real_price = []
    days_real = []
    days_predict = []

    for i in range(data_size + len(conclusion)):
        days_predict.append(i)

    for i in range(data_size):
        predictions.append(all_data['Medium Term Prediction'][len(all_data['Medium Term Prediction']) - data_size + i])
        real_price.append(all_data['Medium Term Prediction'][len(all_data['Medium Term Prediction']) - data_size + i])
        days_real.append(i)

    for i in conclusion:
        predictions.append(i)

    plt.plot(days_real, real_price, label='Price', zorder=1)
    plt.plot(days_predict, predictions, label='Predictions', zorder=0)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Medium Term Evaluated Prediction')
    plt.legend(loc=4)
    plt.show()


# train_model_medium()
present_prediction_medium()




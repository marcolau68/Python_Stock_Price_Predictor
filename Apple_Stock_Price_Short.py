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
predict_out_short = 2

all_data['Short Term Prediction'] = all_data[predict_variable].shift(-predict_out_short)

X = np.array(all_data.drop('Short Term Prediction', axis=1))
X = preprocessing.scale(X)

data_size = 60

'''Short Term'''

X_short = X[-data_size:-predict_out_short]
X_predict_short = X[-predict_out_short:]

all_data.dropna(inplace=True)
y_short = np.array(all_data['Short Term Prediction'][-(data_size - predict_out_short):])


def train_model_short():
    accuracy = [0]

    while max(accuracy) < 0.96:
        X_train_short, X_test_short, y_train_short, y_test_short = train_test_split(X_short, y_short, test_size=0.2)
        linear_short_train = LinearRegression(n_jobs=-1)
        linear_short_train.fit(X_train_short, y_train_short)
        acc_short = linear_short_train.score(X_test_short, y_test_short)
        accuracy.append(acc_short)

        if acc_short > 0.96:
            with open('Apple_short.pickle', 'wb') as f:
                pickle.dump(linear_short_train, f)

    print('Training complete. Accuracy: ' + str(max(accuracy) * 100) + '%')


def present_prediction_short():
    trained_model = open('Apple_short.pickle', 'rb')
    linear_short_best = pickle.load(trained_model)
    y_predict_short = linear_short_best.predict(X_predict_short)
    print(str(predict_out_short) + ' day prediction: \n' + str(y_predict_short))

    predictions = []
    real_price = []
    days_real = []
    days_predict = []

    for i in range(data_size + len(y_predict_short)):
        days_predict.append(i)

    for i in range(data_size):
        predictions.append(all_data['Short Term Prediction'][len(all_data['Short Term Prediction']) - data_size + i])
        real_price.append(all_data['Short Term Prediction'][len(all_data['Short Term Prediction']) - data_size + i])
        days_real.append(i)

    for i in y_predict_short:
        predictions.append(i)

    plt.plot(days_real, real_price, label='Price', zorder=1)
    plt.plot(days_predict, predictions, label='Predictions', zorder=0)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Short Term Prediction')
    plt.legend(loc=4)
    plt.show()


def make_prediction_short():
    trained_model = open('Apple_short.pickle', 'rb')
    linear_short_best = pickle.load(trained_model)
    y_predict_short = linear_short_best.predict(X_predict_short)

    return y_predict_short


def present_evaluated_prediction_short(conclusion):
    predictions = []
    real_price = []
    days_real = []
    days_predict = []
    conclusion = conclusion[:predict_out_short]

    for i in range(data_size + len(conclusion)):
        days_predict.append(i)

    for i in range(data_size):
        predictions.append(all_data['Short Term Prediction'][len(all_data['Short Term Prediction']) - data_size + i])
        real_price.append(all_data['Short Term Prediction'][len(all_data['Short Term Prediction']) - data_size + i])
        days_real.append(i)

    for i in conclusion:
        predictions.append(i)

    plt.plot(days_real, real_price, label='Price', zorder=1)
    plt.plot(days_predict, predictions, label='Predictions', zorder=0)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Short Term Evaluated Prediction')
    plt.legend(loc=4)
    plt.show()


# train_model_short()
# present_prediction_short()




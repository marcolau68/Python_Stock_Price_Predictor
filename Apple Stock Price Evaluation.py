import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle


class Train_stock_model:
    def __init__(self, stock, mode):
        self.stock = stock
        self.data_name = str(self.stock) + '.csv'
        self.model_file = str(self.stock) + '_' + str(mode) + '.pickle'

        self.all_data = pd.read_csv(self.data_name, sep=',', header=0)

        self.all_data = self.all_data[['Open', 'Close', 'Adj Close', 'Volume']]

        self.all_data['%_Range'] = (self.all_data['Close'] - self.all_data['Open']) / self.all_data['Open'] * 100
        self.all_data['Price'] = self.all_data['Adj Close']

        self.all_data = self.all_data[['%_Range', 'Volume', 'Price']]

        self.all_data.fillna(-9999, inplace=True)
        self.all_data.dropna(inplace=True)

        self.predict_variable = 'Price'

        if mode == 'short':
            self.predict_out = 2
            self.data_size = 60
            self.target_acc = 0.96
        elif mode == 'medium':
            self.predict_out = 8
            self.data_size = 600
            self.target_acc = 0.91
        elif mode == 'long':
            self.predict_out = 30
            self.data_size = len(self.all_data['Price'])
            self.target_acc = 0.97

        self.all_data['Prediction'] = self.all_data[self.predict_variable].shift(-self.predict_out)

        self.X = np.array(self.all_data.drop('Prediction', axis=1))
        self.X = preprocessing.scale(self.X)

        self.X_train = self.X[-self.data_size:-self.predict_out]
        self.X_predict = self.X[-self.predict_out:]

        self.all_data.dropna(inplace=True)
        self.y = np.array(self.all_data['Prediction'][-(self.data_size - self.predict_out):])

    def train_model(self):
        accuracy = [0]

        while max(accuracy) < self.target_acc:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
            linear_train = LinearRegression(n_jobs=-1)
            linear_train.fit(X_train, y_train)
            acc = linear_train.score(X_test, y_test)
            accuracy.append(acc)

            if max(accuracy) > self.target_acc:
                with open(self.model_file, 'wb') as f:
                    pickle.dump(linear_train, f)

        print('Training complete. Accuracy: ' + str(max(accuracy) * 100) + '%')

    def make_prediction(self, predict_out):
        self.X_predict = self.X[-predict_out:]

        trained_model = open(self.model_file, 'rb')
        linear = pickle.load(trained_model)
        y_predict_medium = linear.predict(self.X_predict)

        return y_predict_medium

    def evaluated_prediction(self):
        prediction_short = self.make_prediction(2)
        prediction_medium = self.make_prediction(8)
        prediction_long = self.make_prediction(30)

        ws = 0.6
        wm = 0.25
        wl = 0.15

        evaluated_predicted_short = ws * prediction_short[0] + wm * prediction_medium[0] + wl * prediction_long[0]

        wm = 0.65
        wl = 0.35
        evaluated_predicted_medium = []

        for i in range(1, 8):
            evaluated_predicted_medium.append(wm * prediction_medium[i] + wl * prediction_long[i])

        evaluated_predicted_long = []

        for i in range(8, 30):
            evaluated_predicted_long.append(prediction_long[i])

        evaluated_prediction = [evaluated_predicted_short]
        for i in evaluated_predicted_medium:
            evaluated_prediction.append(i)
        for j in evaluated_predicted_long:
            evaluated_prediction.append(j)

        predictions = []
        real_price = []
        days_real = []
        days_predict = []

        for i in range(90):
            days_predict.append(i)

        for i in range(60):
            predictions.append(
                self.all_data['Prediction'][len(self.all_data['Prediction']) - 60 + i])
            real_price.append(
                self.all_data['Prediction'][len(self.all_data['Prediction']) - 60 + i])
            days_real.append(i)

        for i in evaluated_prediction:
            predictions.append(i)

        print(evaluated_prediction)

        plt.plot(days_real, real_price, label='Price', zorder=1)
        plt.plot(days_predict, predictions, label='Predictions', zorder=0)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('30 Days Evaluated Prediction')
        plt.legend(loc=4)
        plt.show()


stock = str(input('Enter stock name \n'))
length = str(input('Enter range for training \n'))

stock_training = Train_stock_model(stock, length)
# stock_training.train_model()
stock_training.evaluated_prediction()




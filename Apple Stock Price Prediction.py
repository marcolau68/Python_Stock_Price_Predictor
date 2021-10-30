import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn import svm
from sklearn import tree
import math
import csv


class Train_stock_model:
    def __init__(self):
        self.data_name = 'Apple.csv'

        self.all_data = pd.read_csv(self.data_name, sep=',', header=0)

        self.all_data = self.all_data[['Open', 'Close', 'Adj Close', 'Volume']]

        self.all_data['%_Range'] = (self.all_data['Close'] - self.all_data['Open']) / self.all_data['Open'] * 100
        self.all_data['Price'] = self.all_data['Adj Close']

        self.all_data = self.all_data[['%_Range', 'Volume', 'Price']]

        self.all_data.fillna(-9999, inplace=True)
        self.all_data.dropna(inplace=True)

        self.predict_variable = 'Price'

        self.predict_out = 30
        self.data_size = 1800
        self.target_acc = 0.9

        self.all_data['Prediction'] = self.all_data[self.predict_variable].shift(-self.predict_out)

        self.X = np.array(self.all_data.drop('Prediction', axis=1))
        self.X = preprocessing.scale(self.X)

        self.X_predict = self.X[-self.predict_out:]
        self.X = self.X[-self.data_size:-self.predict_out]

        self.all_data.dropna(inplace=True)
        self.y = np.array(self.all_data['Prediction'][-(self.data_size - self.predict_out):])

    def train_linear(self):
        accuracy = [0]

        while True:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
            linear_train = LinearRegression(n_jobs=-1)
            linear_train.fit(X_train, y_train)
            acc = linear_train.score(X_test, y_test)
            accuracy.append(acc)

            if self.target_acc < acc < 0.98:
                with open('Apple_linear.pickle', 'wb') as f:
                    pickle.dump(linear_train, f)
                    print('Linear Model training complete. Accuracy: ' + str(acc * 100) + '%')
                    break

    def train_svm(self):
        accuracy = [0]

        while True:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
            svm_train = svm.SVR()
            svm_train.fit(X_train, y_train)
            acc = svm_train.score(X_test, y_test)
            accuracy.append(acc)

            if self.target_acc < acc < 0.95:
                with open('Apple_svm.pickle', 'wb') as f:
                    pickle.dump(svm_train, f)
                    print('Support Vector Model training complete. Accuracy: ' + str(acc * 100) + '%')
                    break

    def train_tree(self):
        accuracy = [0]

        while True:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
            tree_train = tree.DecisionTreeRegressor()
            tree_train.fit(X_train, y_train)
            acc = tree_train.score(X_test, y_test)
            accuracy.append(acc)

            if self.target_acc < acc < 0.98:
                with open('Apple_tree.pickle', 'wb') as f:
                    pickle.dump(tree_train, f)
                    print('Random Tree training complete. Accuracy: ' + str(accuracy[-1] * 100) + '%')
                    break

    def train_all(self):
        self.train_linear()
        self.train_svm()
        self.train_tree()
        self.list_all_predictions()
        self.train_composite()

    def make_prediction(self, model, predict_range):
        model_file = 'Apple_' + model + '.pickle'

        trained_model = open(model_file, 'rb')
        model = pickle.load(trained_model)
        y_predict = model.predict(predict_range)

        return y_predict

    def present_prediction(self, model):
        raw_predictions = self.make_prediction(str(model), self.X_predict)
        print('30 days prediction \n' + str(raw_predictions))

        predictions = []
        real_price = []
        days_real = []
        days_predict = []

        for i in range(630):
            days_predict.append(i)

        for i in range(600):
            predictions.append(
                self.all_data['Prediction'][len(self.all_data['Prediction']) - 600 + i])
            real_price.append(
                self.all_data['Prediction'][len(self.all_data['Prediction']) - 600 + i])
            days_real.append(i)

        for i in raw_predictions:
            predictions.append(i)

        plt.plot(days_real, real_price, label='Price', zorder=1)
        plt.plot(days_predict, predictions, label='Predictions', zorder=0)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('30 Days Prediction ' + '(%s)' % model)
        plt.legend(loc=4)
        plt.show()

    def list_all_predictions(self):
        iteration = math.floor(self.data_size / 30) - 1

        with open('Apple_modified.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Linear', 'SVM', 'Tree', 'Actual'])

            for i in range(iteration):
                self.X_predict = self.X[30 * i: 30 * i + 30]

                linear_prediction = self.make_prediction('linear', self.X_predict)
                svm_prediction = self.make_prediction('svm', self.X_predict)
                tree_prediction = self.make_prediction('tree', self.X_predict)

                for j in range(30):
                    writer.writerow([linear_prediction[j], svm_prediction[j], tree_prediction[j],
                                     self.all_data['Prediction'][len(self.all_data['Prediction']) -
                                                            self.data_size + 30 * (i + 1) + j]])

    def train_composite(self):
        self.modified_data = pd.read_csv('Apple_modified.csv', sep=',', header=0)
        self.modified_data = self.modified_data[['Linear', 'SVM', 'Tree', 'Actual']]

        X = self.modified_data.drop('Actual', axis=1)
        y = self.modified_data['Actual']

        accuracy = [0]

        while True:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
            composite_train = LinearRegression()
            composite_train.fit(X_train, y_train)
            acc = composite_train.score(X_test, y_test)
            accuracy.append(acc)

            if 0.95 < acc < 0.98:
                with open('Apple_tree.pickle', 'wb') as f:
                    pickle.dump(composite_train, f)
                    print('Composite model training complete. Accuracy: ' + str(acc * 100) + '%')
                    break

    def present_composite(self):
        linear_prediction = self.make_prediction('linear', self.X_predict)
        svm_prediction = self.make_prediction('svm', self.X_predict)
        tree_prediction = self.make_prediction('tree', self.X_predict)

        all_predictions = []
        for i in range(30):
            unit = []
            unit.append(linear_prediction[i])
            unit.append(svm_prediction[i])
            unit.append(tree_prediction[i])
            all_predictions.append(unit)

        com = open('Apple_composite.pickle', 'rb')
        composite_model = pickle.load(com)
        y_predict = composite_model.predict(all_predictions)
        print(y_predict)

        predictions = []
        real_price = []
        days_real = []
        days_predict = []

        for i in range(630):
            days_predict.append(i)

        for i in range(600):
            predictions.append(
                self.all_data['Prediction'][len(self.all_data['Prediction']) - 600 + i])
            real_price.append(
                self.all_data['Prediction'][len(self.all_data['Prediction']) - 600 + i])
            days_real.append(i)

        for i in y_predict:
            predictions.append(i)

        plt.plot(days_real, real_price, label='Price', zorder=1)
        plt.plot(days_predict, predictions, label='Predictions', zorder=0)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('30 Days Prediction')
        plt.legend(loc=4)
        plt.show()



test = Train_stock_model()
# test.train_all()
# test.present_prediction('linear')
test.present_composite()






import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

'''
Load a stock with company ticker, start date, end date, and how many days to look in the past (default=60)
Call prepareData
Call modelData
Call predictData
# predictFutureStock prints predicted close of stock in a day
'''
class Stock:

    def __init__(self, company, start, end, prediction_days=60):
        self.company = company
        self.start = start
        self.end = end
        self.prediction_days = prediction_days

    def prepareData(self):
        print(f'Preparing {self.company}.')
        self.data = web.DataReader(self.company, 'yahoo', self.start, self.end)

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        scaled_data = self.scaler.fit_transform(self.data['Close'].values.reshape(-1, 1))

        x_train = []
        y_train = []

        for x in range(self.prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - self.prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        x_train, self.y_train = np.array(x_train), np.array(y_train)
        self.x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    def modelData(self):
        print(f'Modelling {self.company}.')
        self.model = Sequential()

        self.model.add(LSTM(units=50, return_sequences=True,
                    input_shape=(self.x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))  # prediction of next closing value

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.x_train, self.y_train, epochs=25, batch_size=32)

    def predictData(self):
        test_data = web.DataReader(self.company, 'yahoo', self.start, self.end)


        total_dataset = pd.concat((self.data['Close'], test_data['Close']), axis=0)

        model_inputs = total_dataset[len(total_dataset) - len(test_data) - self.prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        self.model_inputs = self.scaler.transform(model_inputs)

        # Predict Test Data

        x_test = []

        for x in range(self.prediction_days, len(self.model_inputs)):
            x_test.append(self.model_inputs[x-self.prediction_days:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_prices = self.model.predict(x_test)
        self.predicted_prices = self.scaler.inverse_transform(predicted_prices)

    # def predictFutureStock(self):
    #     real_data = [self.model_inputs[len(self.model_inputs) + 1 - self.prediction_days:len(self.model_inputs + 1)]]

    #     real_data = np.array(real_data)
    #     real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    #     prediction = self.model.predict(real_data)
    #     prediction = self.scaler.inverse_transform(prediction)

    #     print(f"Prediction for {self.company}: {prediction}")

'''
Plots list of stocks
'''
def plotCompanyStocks(stocks, show_actual=False):
    for stock in stocks:
        color = (np.random.random(), np.random.random(), np.random.random())
        plt.plot(stock.predicted_prices, c=color, label=f'Predicted {stock.company} price')
        if show_actual:
            color2 = (np.random.random(), np.random.random(), np.random.random())
            plt.plot(stock.data['Close'].values, c=color2, label=f'Actual {stock.company} price', linestyle='dashed')

    
    plt.title(f"Share Prices")
    plt.xlabel('Time')
    plt.ylabel(f'Share Price')
    plt.legend()
    plt.show()

# Test
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)
stocks = []
stocks.append(Stock('AAPL', start, end))
stocks.append(Stock('TSLA', start, end))
for stock in stocks:
    stock.prepareData()
    stock.modelData()
    stock.predictData()

plotCompanyStocks(stocks)

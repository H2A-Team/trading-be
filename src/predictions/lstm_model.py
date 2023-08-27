import os

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

from settings import settings

TRAIN_SIZE = 0.8
STEPS = 60


def split_dataset(dataset, train_size=TRAIN_SIZE):
    index = int(len(dataset) * train_size)
    return dataset[:index], dataset[index:]


def build_model_filename(path=settings.PREDICTION_MODEL_LOCATION) -> str:
    prefix = path if path.endswith('/') else path + '/'
    return f'{prefix}lstm.keras'


class LSTMModel:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        # self.filtered_dataset = pd.DataFrame(data=dataset["Close"].to_numpy(), index=dataset["Date"], columns=["Close"])
        self.filtered_dataset = pd.DataFrame(data=dataset["Close"].to_numpy(), columns=["Close"])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self.load_or_train_model()

    # example of interval values: 1min
    def apply_closing_price_indicator(self, n_future_preds: int):
        x_train_data, y_train_data = self._normalize_dataset()
        self.model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose="2")

        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)

        predictions = self._moving_test_window_preds(test_dataset, test_dataset.shape[0] - STEPS, n_future_preds)
        return predictions.flatten().tolist()

    def predict_next_candle(self, open_df, high_df, low_df, close_df):
        # open
        self.dataset = open_df
        self.filtered_dataset = open_df
        x_train_data, y_train_data = self._normalize_dataset()
        self.model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose="2")
        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)
        open_predictions = self._moving_test_window_preds(test_dataset, test_dataset.shape[0] - STEPS, 1)

        # high
        self.dataset = high_df
        self.filtered_dataset = high_df
        x_train_data, y_train_data = self._normalize_dataset()
        self.model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose="2")
        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)
        high_predictions = self._moving_test_window_preds(test_dataset, test_dataset.shape[0] - STEPS, 1)

        # low
        self.dataset = low_df
        self.filtered_dataset = low_df
        x_train_data, y_train_data = self._normalize_dataset()
        self.model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose="2")
        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)
        low_predictions = self._moving_test_window_preds(test_dataset, test_dataset.shape[0] - STEPS, 1)

        # close
        self.dataset = close_df
        self.filtered_dataset = close_df
        x_train_data, y_train_data = self._normalize_dataset()
        self.model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose="2")
        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)
        close_predictions = self._moving_test_window_preds(test_dataset, test_dataset.shape[0] - STEPS, 1)

        return (
            open_predictions.flatten().tolist(),
            high_predictions.flatten().tolist(),
            low_predictions.flatten().tolist(),
            close_predictions.flatten().tolist(),
        )

    def _normalize_dataset(self):
        final_dataset = self.filtered_dataset.values
        train_data, _ = split_dataset(final_dataset)

        scaled_data = self.scaler.fit_transform(final_dataset)

        x_train_data, y_train_data = [], []
        for i in range(STEPS, len(train_data)):
            x_train_data.append(scaled_data[i - STEPS : i, 0])
            y_train_data.append(scaled_data[i, 0])

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        return x_train_data, y_train_data

    def _moving_test_window_preds(self, dataset, start, n_future_preds):
        # Declare variable where we store the prediction made on each window and the moving input window
        predictions = []
        moving_input_window = []

        # Set the inputs
        inputs = self.scaler.transform(dataset.values)

        moving_input_window.append(inputs[start : (start + STEPS), :])
        moving_input_window = np.array(moving_input_window)
        moving_input_window = np.reshape(
            moving_input_window, (moving_input_window.shape[0], moving_input_window.shape[1], 1)
        )

        # Loop over the amount of future predictions we want to make
        for i in range(n_future_preds):
            # Predict the next price based on the moving_input_window
            y_hat = self.model.predict(moving_input_window)

            # Append y_hat to predictions
            predictions.append(y_hat[0, :])

            # Reshape y_hat for concatenation with moving test window
            y_hat = y_hat.reshape(1, 1, 1)

            # Remove first element
            moving_input_window = np.concatenate((moving_input_window[:, 1:, :], y_hat), axis=1)

        predictions = self.scaler.inverse_transform(pd.DataFrame(predictions))
        return predictions

    def load_or_train_model(self, restrict_train=False):
        model_filename = build_model_filename()

        if restrict_train or not os.path.isfile(model_filename):
            # normalize dataset and train lstm model
            x_train_data, y_train_data = self._normalize_dataset()
            model = self._train(x_train_data, y_train_data)
            model.save(model_filename)
        else:
            # use prebuilt model
            model = load_model(model_filename)

        self.model = model
        return model

    def _train(self, x_train_data, y_train_data):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose="2")

        return model

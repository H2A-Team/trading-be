import os

import numpy as np
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import Sequential, load_model
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from settings import settings

TRAIN_SIZE = 0.7
STEPS = 50


def split_dataset(dataset, train_size=TRAIN_SIZE):
    index = int(len(dataset) * train_size)
    return dataset[:index], dataset[index:]


def build_model_filename(path=settings.PREDICTION_MODEL_LOCATION) -> str:
    return os.path.join(path, "rnn.keras")


class SimpleRNNModel:
    def __init__(self, dataset: DataFrame):
        self.dataset = dataset
        self.filtered_dataset = DataFrame(data=dataset["Close"].to_numpy(), columns=["Close"])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._load_or_train_model()

    # example of interval values: 1min
    def apply_closing_price_indicator(self, n_future_preds: int):
        x_train_data, y_train_data = self._normalize_dataset()
        self.model.fit(x_train_data, y_train_data, epochs=50, batch_size=32, verbose="2")

        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)

        predictions = self._moving_test_window_preds(test_dataset, test_dataset.shape[0] - STEPS, n_future_preds)
        return predictions.flatten().tolist()

    def predict_next_candle(self, open_df, high_df, low_df, close_df):
        # open
        self.dataset = open_df
        self.filtered_dataset = open_df
        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)
        open_predictions = self._moving_test_window_preds(test_dataset, test_dataset.shape[0] - STEPS, 1)

        # high
        self.dataset = high_df
        self.filtered_dataset = high_df
        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)
        high_predictions = self._moving_test_window_preds(test_dataset, test_dataset.shape[0] - STEPS, 1)

        # low
        self.dataset = low_df
        self.filtered_dataset = low_df
        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)
        low_predictions = self._moving_test_window_preds(test_dataset, test_dataset.shape[0] - STEPS, 1)

        # close
        self.dataset = close_df
        self.filtered_dataset = close_df
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

        predictions = self.scaler.inverse_transform(DataFrame(predictions))
        return predictions

    def _load_or_train_model(self):
        model_filename = build_model_filename()

        if os.path.isfile(model_filename):
            # use prebuilt model
            model = load_model(model_filename)
        else:
            # normalize dataset and train lstm model
            x_train_data, y_train_data = self._normalize_dataset()
            model = self._train(x_train_data, y_train_data)
            model.save(model_filename)

        self.model = model
        return model

    def _train(self, x_train_data, y_train_data):
        model = Sequential()

        # adding first RNN layer and dropout regulatization
        model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        model.add(Dropout(0.2))

        # adding second RNN layer and dropout regulatization
        model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
        model.add(Dropout(0.2))

        # adding third RNN layer and dropout regulatization
        model.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
        model.add(Dropout(0.2))

        # adding fourth RNN layer and dropout regulatization
        model.add(SimpleRNN(units=50))
        model.add(Dropout(0.2))

        # adding the output layer
        model.add(Dense(units=1))

        # compiling RNN
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

        # fitting the RNN
        model.fit(x_train_data, y_train_data, epochs=50, batch_size=32)

        return model

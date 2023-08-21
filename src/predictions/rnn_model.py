import os

import numpy as np
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import Sequential, load_model
from pandas import DataFrame, concat, date_range
from sklearn.preprocessing import MinMaxScaler

from settings import settings

TRAIN_SIZE = 0.7
STEPS = 60


def split_dataset(dataset, train_size=TRAIN_SIZE):
    index = int(len(dataset) * train_size)
    return dataset[:index], dataset[index:]


def build_model_filename(path=settings.PREDICTION_MODEL_LOCATION) -> str:
    prefix = path if path.endswith('/') else path + '/'
    return f'{prefix}rnn.keras'


class SimpleRNNModel:
    def __init__(self, dataset: DataFrame):
        self.dataset = dataset
        self.filtered_dataset = DataFrame(data=dataset["Close"].to_numpy(), index=dataset["Date"], columns=["Close"])
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._load_or_train_model()

    # example of interval values: 1min
    def predict_future_prices(self, n_future_preds: int, interval: str):
        train_dataset, test_dataset = split_dataset(self.filtered_dataset, TRAIN_SIZE)
        self.scaler.fit(train_dataset.values)

        future_prediction = self._moving_test_window_preds(
            test_dataset,
            test_dataset.shape[0] - STEPS,
            n_future_preds,
        )
        future_prediction_df = DataFrame(
            data={'Predictions': future_prediction.flatten()},
            index=date_range(
                start=test_dataset.index[test_dataset.shape[0] - 1],
                periods=n_future_preds + 1,
                freq=interval,
            )[1:],
        )

        test_dataset = concat([test_dataset, future_prediction_df])
        return test_dataset

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

    def _moving_test_window_preds(self, test_dataset, start, n_future_preds):
        # Declare variable where we store the prediction made on each window and the moving input window
        prediction = []
        moving_input_window = []

        # Set the inputs
        inputs = self.scaler.transform(test_dataset.values)

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
            prediction.append(y_hat[0, :])

            # Reshape y_hat for concatenation with moving test window
            y_hat = y_hat.reshape(1, 1, 1)

            # Remove first element
            moving_input_window = np.concatenate((moving_input_window[:, 1:, :], y_hat), axis=1)

        prediction = self.scaler.inverse_transform(DataFrame(prediction))
        return prediction

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

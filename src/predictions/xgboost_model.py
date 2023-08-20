import os

from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from settings import settings

TRAIN_SIZE = 0.7
VALID_SIZE = 0.15


def split_dataset(dataset, train_size=TRAIN_SIZE, valid_size=VALID_SIZE):
    last_train_index = int(len(dataset) * train_size)
    last_valid_index = int(len(dataset) * (train_size + valid_size))

    train_dataset = dataset[:last_train_index]
    valid_dataset = dataset[last_train_index:last_valid_index]
    test_dataset = dataset[last_valid_index:]

    return train_dataset, valid_dataset, test_dataset


def build_model_filename(path=settings.PREDICTION_MODEL_LOCATION) -> str:
    prefix = path if path.endswith('/') else path + '/'
    return f'{prefix}xgboost.json'


class XGBoostModel:
    def __init__(self, dataset: DataFrame):
        self.dataset = dataset
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._load_or_train_model()

    # example of interval values: 1min
    def predict_future_prices(self):
        train_dataset, _, test_dataset = split_dataset(self.dataset, TRAIN_SIZE, VALID_SIZE)
        self.scaler.fit(train_dataset.values)

        # Because of moving averages and MACD line
        self.filtered_dataset = self.filtered_dataset.iloc[33:]
        self.filtered_dataset.index = range(len(self.filtered_dataset))

        # y_test_dataset = test_dataset['Close'].copy()
        # x_test_dataset = test_dataset.drop(columns=['Close'], axis=1)

        return test_dataset

    def _apply_moving_averages_indicator(self):
        self.dataset['EMA_9'] = self.dataset['Close'].ewm(9).mean().shift()
        self.dataset['SMA_5'] = self.dataset['Close'].rolling(5).mean().shift()
        self.dataset['SMA_10'] = self.dataset['Close'].rolling(10).mean().shift()
        self.dataset['SMA_15'] = self.dataset['Close'].rolling(15).mean().shift()
        self.dataset['SMA_30'] = self.dataset['Close'].rolling(30).mean().shift()

    def _apply_relative_strength_index_indicator(self, n=14):
        close = self.dataset['Close']
        delta = close.diff()
        delta = delta[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp < 0] = 0
        pricesDown[pricesDown > 0] = 0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp / rollDown
        rsi = 100.0 - (100.0 / (1.0 + rs))

        self.dataset['RSI'] = rsi.fillna(0)

    def _load_or_train_model(self, is_applied_indicators=True):
        model_filename = build_model_filename()

        if os.path.isfile(model_filename):
            # use prebuilt model
            model = XGBRegressor(objective='reg:squarederror')
            model = model.load_model(model_filename)
        else:
            if not is_applied_indicators:
                self._apply_moving_averages_indicator()
                self._apply_relative_strength_index_indicator()

            train_dataset, _, _ = split_dataset(self.dataset, TRAIN_SIZE, VALID_SIZE)
            x_train_data = train_dataset.drop(columns=["Close"], axis=1)
            y_train_data = train_dataset["Close"]

            model = self._train(x_train_data, y_train_data)
            model.save_model(model_filename)

        self.model = model
        return model

    def _train(self, x_train_data, y_train_data):
        model = XGBRegressor(objective='reg:squarederror')
        model.fit(x_train_data, y_train_data)
        return model

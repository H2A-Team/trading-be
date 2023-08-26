import os

import pandas as pd
from xgboost import XGBRegressor

from settings import settings

TRAIN_SIZE = 0.8

def split_dataset(dataset, train_size=TRAIN_SIZE):
    index = int(len(dataset) * train_size)
    return dataset[:index], dataset[index:]


def build_model_filename(path=settings.PREDICTION_MODEL_LOCATION) -> str:
    prefix = path if path.endswith("/") else path + "/"
    return f"{prefix}xgboost.json"


class XGBoostModel:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.model = self._load_or_train_model()

    # example of interval values: "1min"
    def predict_future_prices(self, n_future_preds: int):
        dataset = self.dataset.copy()
        dataset = self._apply_moving_averages_indicator(dataset)
        dataset = self._apply_relative_strength_index_indicator(dataset)

        data_train = dataset.iloc[30:, :]
        data_train.index = range(len(data_train))

        xtrain = data_train.drop(columns=["Close"], axis=1)
        ytrain = data_train["Close"].copy()
        self.model.fit(xtrain, ytrain, eval_set=[(xtrain, ytrain)], verbose=False)

        predictions = []
        for i in range(n_future_preds):
            xtest = xtrain.iloc[-1:, :]

            # predict
            yhat = self.model.predict(xtest)
            predictions.append(yhat[0])

            # featuring
            dataset = pd.concat([dataset, pd.DataFrame(data={"Close": [yhat[0]]}, index=[dataset.index[-1] + 1])])
            dataset = self._apply_moving_averages_indicator(dataset)
            dataset = self._apply_relative_strength_index_indicator(dataset)

            pred_df = dataset.tail(1)
            pred_df.index = xtrain.tail(1).index
            xtrain = pd.concat([xtrain.iloc[1:, :], pred_df.drop(columns=["Close"], axis=1)])

        return predictions

    def _apply_moving_averages_indicator(self, dataset):
        dataset["EMA_9"] = dataset["Close"].ewm(9).mean().shift()
        dataset["SMA_5"] = dataset["Close"].rolling(5).mean().shift()
        dataset["SMA_10"] = dataset["Close"].rolling(10).mean().shift()
        dataset["SMA_15"] = dataset["Close"].rolling(15).mean().shift()
        dataset["SMA_30"] = dataset["Close"].rolling(30).mean().shift()
        return dataset

    def _apply_relative_strength_index_indicator(self, dataset, n=14):
        close = self.dataset["Close"]
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

        dataset["RSI"] = rsi.fillna(0)
        return dataset

    def _load_or_train_model(self, is_applied_indicators=True):
        model_filename = build_model_filename()

        if os.path.isfile(model_filename):
            # use prebuilt model
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=1000,
                max_depth=5,
                min_child_weight=2,
                learning_rate=0.3,
                early_stopping_rounds=20,
            )
            model = model.load_model(model_filename)
        else:
            # if not is_applied_indicators:
            #     self._apply_moving_averages_indicator()
            #     self._apply_relative_strength_index_indicator()

            train_dataset, test_dataset = split_dataset(self.dataset, TRAIN_SIZE)
            x_train_data = train_dataset.drop(columns=["Close"], axis=1)
            y_train_data = train_dataset["Close"]
            x_test_data = test_dataset.drop(columns=["Close"], axis=1)
            y_test_data = test_dataset["Close"]

            model = self._train(x_train_data, y_train_data, x_test_data, y_test_data)
            model.save_model(model_filename)

        self.model = model
        return model

    def _train(self, x_train_data, y_train_data, x_test_data, y_test_data):
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            max_depth=5,
            min_child_weight=2,
            learning_rate=0.3,
            early_stopping_rounds=20,
        )
        model.fit(
            x_train_data,
            y_train_data,
            eval_set=[(x_train_data, y_train_data), (x_test_data, y_test_data)],
            verbose=False,
        )
        return model

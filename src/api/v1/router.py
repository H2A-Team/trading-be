import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from pyti.rate_of_change import rate_of_change
from pyti.relative_strength_index import relative_strength_index

from models.candle import Candle
from models.crypto_symbol import CryptoSymbol
from models.response import RestResponseList
from models.timeframe_prediction_body import TimeframePredictionBody
from predictions.lstm_model import LSTMModel
from predictions.rnn_model import SimpleRNNModel
from predictions.xgboost_model import XGBoostModel
from services.binance_rest_service import BinanceRestService
from settings import settings

router = APIRouter(prefix="/v1")
service = BinanceRestService(settings.BINANCE_REST_API_URL, settings.BINANCE_REST_API_KEY)


@router.get(
    path="/symbols",
    response_model=RestResponseList[CryptoSymbol],
    status_code=status.HTTP_200_OK,
    tags=["Symbol"],
)
async def get_symbols(symbols: list[str] = Query(default=None)):
    if symbols is None or len(symbols) == 0:
        symbols = settings.BINANCE_MARKET_SYMBOLS
    else:
        for i in range(0, len(symbols)):
            symbols[i] = symbols[i].upper()

    status, exchange_info = service.get_exchange_information(symbols)
    if status == 200:
        crypto_symbols = list(
            map(
                lambda item: CryptoSymbol(
                    symbol=item["symbol"],
                    status=item["status"],
                    baseAsset=item["baseAsset"],
                    baseAssetPrecision=item["baseAssetPrecision"],
                    quoteAsset=item["quoteAsset"],
                    quotePrecision=item["quotePrecision"],
                    quoteAssetPrecision=item["quoteAssetPrecision"],
                    baseCommissionPrecision=item["baseCommissionPrecision"],
                    quoteCommissionPrecision=item["quoteCommissionPrecision"],
                ),
                exchange_info,
            )
        )

        total = len(crypto_symbols)
        limit = total if total > 20 else 20
        return RestResponseList(data=crypto_symbols, total=total, offset=0, limit=limit)

    raise HTTPException(status_code=500)


@router.get(
    path="/symbols/{symbol}/candles",
    response_model=RestResponseList[Candle],
    status_code=status.HTTP_200_OK,
    tags=["Symbol"],
)
async def get_candles_by_interval(symbol: str, interval: str):
    symbol = symbol.upper()
    if symbol not in settings.BINANCE_MARKET_SYMBOLS or interval not in settings.BINANCE_MARKET_INTERVALS:
        return RestResponseList(data=[], total=0, offset=0, limit=1000)

    max_candles = settings.BINANCE_MARKET_MAX_CANDLES[interval]
    limit = 1000 if max_candles >= 1000 else max_candles
    total_ui_klines = []
    end_time = None

    # fetch by descending time
    while limit > 0:
        _, ui_klines = service.get_ui_klines(symbol, interval, limit=limit, end_time=end_time)

        # time(ui_klines) is before time(total_ui_klines)
        total_ui_klines = ui_klines + total_ui_klines
        if len(total_ui_klines) == 0:
            break

        end_time = total_ui_klines[0][0] - 1  # oldest_start_time - 1

        # recalculate limit
        max_candles -= limit
        limit = 1000 if max_candles >= 1000 else max_candles

    candles = list(
        map(
            lambda kline: Candle(
                binanceEventTimestamp=kline[0],
                symbol=symbol,
                startIntervalTimestamp=kline[0],
                endIntervalTimestamp=kline[6],
                interval=interval,
                openPrice=kline[1],
                closePrice=kline[4],
                highPrice=kline[2],
                lowPrice=kline[3],
                volume=kline[5],
            ),
            total_ui_klines,
        )
    )

    num_of_candles = len(candles)
    resp_limit = max(settings.BINANCE_MARKET_MAX_CANDLES[interval], num_of_candles)
    return RestResponseList(data=candles, total=num_of_candles, offset=0, limit=resp_limit)


@router.post(
    path="/symbols/{symbol}/predict-timeframe",
    status_code=status.HTTP_200_OK,
    tags=["Symbol"],
)
async def predict_future_price_by_interval(symbol: str, body: TimeframePredictionBody):
    interval = body.interval
    model_name = body.model
    symbol = symbol.upper()
    indicator_types = list(map(lambda it: it.lower(), body.indicatorTypes))
    if (
        symbol not in settings.BINANCE_MARKET_SYMBOLS
        or interval not in settings.BINANCE_MARKET_INTERVALS
        or len(indicator_types) == 0
        or model_name not in settings.BINANCE_PREDICTION_MODELS
        or body.rocLength >= 1
        or body.rsiLength >= 1
    ):
        return {}

    for it in indicator_types:
        if it not in settings.BINANCE_PREDICTION_INDICATORS:
            return {}

    max_candles = settings.BINANCE_MARKET_MAX_CANDLES[interval]
    limit = 1000 if max_candles >= 1000 else max_candles
    total_ui_klines = []
    end_time = None

    # fetch by descending time
    while limit > 0:
        _, ui_klines = service.get_ui_klines(symbol, interval, limit=limit, end_time=end_time)

        # time(ui_klines) is before time(total_ui_klines)
        total_ui_klines = ui_klines + total_ui_klines
        if len(total_ui_klines) == 0:
            break

        end_time = total_ui_klines[0][0] - 1  # oldest_start_time - 1

        # recalculate limit
        max_candles -= limit
        limit = 1000 if max_candles >= 1000 else max_candles

    timestamps = []
    close_prices = []
    for kline in total_ui_klines:
        timestamps.append(kline[0])
        close_prices.append(float(kline[4]))

    df = pd.DataFrame({"Close": close_prices})
    if model_name == "lstm":
        model = LSTMModel(df)
    elif model_name == "rnn":
        model = SimpleRNNModel(df)
    else:
        model = XGBoostModel(df)

    predictions = model.apply_closing_price_indicator(len(df))
    roc = rsi = None
    if "roc" in indicator_types:
        roc = np.nan_to_num(rate_of_change(predictions, body.rocLength + 1))
    if "rsi" in indicator_types:
        rsi = np.nan_to_num(relative_strength_index(predictions, body.rsiLength))

    result = {}
    if "close" in indicator_types:
        result["close"] = [{"timestamp": timestamps[i], "value": predictions[i]} for i in range(len(timestamps))]
    if "roc" in indicator_types:
        result["roc"] = [{"timestamp": timestamps[i], "value": roc[i]} for i in range(len(timestamps))]
    if "rsi" in indicator_types:
        result["rsi"] = [{"timestamp": timestamps[i], "value": rsi[i]} for i in range(len(timestamps))]

    return result

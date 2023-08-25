from fastapi import APIRouter, HTTPException, Query, status

from models.candle import Candle
from models.crypto_symbol import CryptoSymbol
from models.response import RestResponseList
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

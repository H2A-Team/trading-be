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
    path="/symbols/{symbol}/complete_candles",
    response_model=RestResponseList[Candle],
    status_code=status.HTTP_200_OK,
    tags=["Symbol"],
)
async def get_candles_by_timeframe(symbol: str, interval: str):
    if symbol.upper() not in settings.BINANCE_MARKET_SYMBOLS or interval not in settings.BINANCE_MARKET_INTERVALS:
        return RestResponseList(data=[], total=0, offset=0, limit=1000)

    _, ui_klines = service.get_ui_klines(symbol, interval, limit=1000)
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
            ui_klines,
        )
    )

    return RestResponseList(data=candles, total=len(candles), offset=0, limit=1000)

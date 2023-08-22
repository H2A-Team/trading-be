from fastapi import APIRouter, HTTPException, Query, status

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

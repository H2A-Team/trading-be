import requests
from typing import List, Union


def concat_base_url_and_path(base_url: str, endpoint: str):
    return base_url.removesuffix("/") + "/" + endpoint.removeprefix("/")


class BinanceRestService:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def get_exchange_information(self, symbols: Union[List[str], None] = None):
        status_code, data = self._get("exchangeInfo", params={"symbols": symbols})
        result = data["symbols"] if status_code == 200 else "ERROR"
        return status_code, result

    # params: query parameters, data: request body
    def _request(self, method: str, endpoint: str, params=None, data=None):
        url = concat_base_url_and_path(self.base_url, endpoint)
        headers = {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json",
        }

        resp = requests.request(method, url, params=params, data=data, headers=headers)
        return resp.status_code, resp.json()

    def _get(self, endpoint: str, params=None):
        return self._request("GET", endpoint, params=params)

    def _post(self, endpoint: str, data=None):
        return self._request("POST", endpoint, data=data)

    def _put(self, endpoint: str, data=None):
        return self._request("PUT", endpoint, data=data)

    def _patch(self, endpoint: str, data=None):
        return self._request("PATCH", endpoint, data=data)

    def _delete(self, endpoint: str, data=None):
        return self._request("DELETE", endpoint, data=data)

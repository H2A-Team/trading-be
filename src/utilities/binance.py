def create_channel_name(interval):
    return "kline_" + interval


def create_stream_name(symbol, interval):
    return symbol.lower() + "@" + create_channel_name(interval)


def create_combination_stream_names(symbols, intervals):
    stream_names = []
    for symbol in symbols:
        for interval in intervals:
            stream_names.append(create_stream_name(symbol, interval))
    return stream_names


def normalize_stream_name(stream_name: str):
    idx = stream_name.find("@kline_")
    return stream_name[0:idx].lower() + stream_name[idx:] if idx != -1 else stream_name

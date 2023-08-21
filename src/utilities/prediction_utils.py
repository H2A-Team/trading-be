# from pandas import DataFrame, RangeIndex, to_datetime


# def normalize_dataset(dataset: DataFrame) -> DataFrame:
#     dataset.insert(0, "Date", dataset.index)
#     dataset.index = RangeIndex(0, len(dataset), 1)
#     dataset['Date'] = to_datetime(dataset["Date"], format='%Y-%m-%d')
#     return dataset.sort_values(by='Date', ascending=True, axis=0)

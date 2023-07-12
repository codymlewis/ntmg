"""
A fast and simple data management library for machine learning
"""

from typing import Dict, Callable, Iterable, Self
import numpy as np
from numpy.typing import NDArray


class DatasetDict:
    def __init__(self, data: Dict[str, NDArray]):
        self.__data = data
        length = 0
        for k, v in data.items():
            if length == 0:
                length = len(v)
            elif length != len(v):
                raise AttributeError(f"Data should be composed of equal length arrays, column {k} has length {len(v)} should be {length}")
        self.length = length

    def select(self, idx: int | Iterable[int | bool]):
        return DatasetDict({k: v[idx] for k, v in self.__data.items()})

    def map(self, mapping_fn: Callable[[Dict[str, NDArray]], Dict[str, NDArray]]) -> Self:
        self.__data = mapping_fn(self.__data)
        return self

    def __getitem__(self, i: str) -> NDArray:
        return self.__data[i]
    
    def __len__(self) -> int:
        return len(self.__data['X'])
    
    def __str__(self) -> str:
        return str(self.__data)
    
    def short_details(self) -> str:
        details = "{"
        for k, v in self.__data.items():
            details += f"{k}: type {v.dtype}, shape {v.shape}, range [{v.min()}, {v.max()}], "
        details = details[:-2] + "}"
        return details


class Dataset:
    def __init__(self, data: Dict[str, Dict[str, NDArray] | DatasetDict]):
        # print(data)
        if np.all([isinstance(v, DatasetDict) for v in data.values()]):
            self.__data = data
        else:
            self.__data = {k: DatasetDict(v) for k, v in data.items()}
    
    def __getitem__(self, i: str) -> DatasetDict:
        return self.__data[i]


    def __str__(self) -> str:
        string = "{\n"
        for k, v in self.__data.items():
            string += f"\t{k}: {v.short_details()}\n"
        string += "}"
        return string
    
    def map(self, mapping_fn: Callable[[Dict[str, NDArray]], Dict[str, NDArray]]) -> Self:
        for v in self.__data.values():
            v.map(mapping_fn)
        return self
    
    def keys(self) -> Iterable[str]:
        return self.__data.keys()

    def select(self, idx_dict: Dict[str, int | Iterable[int | bool]]):
        return Dataset({k: self.__data[k].select(idx) for k, idx in idx_dict.items()})
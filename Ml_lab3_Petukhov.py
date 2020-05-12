# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:32:01 2020

@author: Petukhov Alexandr Sergeevich https://github.com/Tabaki59/ML-Labs
"""
# Задание 1 (готовим данные, считаем метрики)
import numpy as np
import pandas as pd 
import json # чтоб сделать из json Фрейм
from pandas.io.json import json_normalize # чтоб нормализовать json файл


def cut_for_speed(obj): # Отрезаем 10 процентов строк для ускорения
    msk = np.random.rand(len(obj)) < 0.1
    new_obj = obj[msk]
    return new_obj

def json_convert(df, JSON_COLUMNS): # Тут мы преобразуем все json столбцы в нормальные и красивые и чтоб окей было
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
data = pd.read_csv(r"G:\Учебка\Машинка\train.csv", sep = ",",  dtype = {'fullVisitorId': 'str'},
                   converters={column: json.loads for column in JSON_COLUMNS}, 
                   nrows = None)
data = json_convert(data, JSON_COLUMNS) # Зовем функцию разобраться до конца с json
data = cut_for_speed(data) # Отрезаем 10 процентов строк для ускорения (Точнее зовем функцию обрезания)

#  Удалить колонки с не доступно в демо версии, очистить nan'ы


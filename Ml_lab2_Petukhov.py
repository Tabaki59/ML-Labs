# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:51:24 2020

@author:  Petukhov Alexandr Sergeevich https://github.com/Tabaki59/ML-Labs
"""

import math
import numpy as np
import pandas as pd 

data = pd.read_csv(r"G:\Учебка\Анализ данных\steam.csv", sep=',')

# Очистка данных
data = data.dropna() 
data = data.drop(columns=["appid","name","release_date","developer","publisher","platforms","categories","genres",
                          "steamspy_tags","average_playtime","owners",]) 
data = data[(data["median_playtime"] != 0)]
data = data[(data["price"] != 0)]

# Удаление выбросов по колонке время в игре
def remove_outliers_for_playtime(data):
    '''Избавляемся от выбросов для зависимой переменной'''
        # Calculate first and third quartile
    first_quartile = data["median_playtime"].describe()['25%']
    third_quartile = data["median_playtime"].describe()['75%']
            
        # Interquartile range
    iqr = third_quartile - first_quartile
            
        # Remove outliers
    data = data[(data["median_playtime"] > (first_quartile - 3 * iqr)) &
                (data["median_playtime"] < (third_quartile + 3 * iqr))]
    return data

data = remove_outliers_for_playtime(data)


# Непосредственно лабортаторка
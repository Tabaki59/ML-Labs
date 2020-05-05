# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:51:24 2020

@author:  Petukhov Alexandr Sergeevich https://github.com/Tabaki59/ML-Labs
"""

import math
import numpy as np
import pandas as pd 
from pandas import DataFrame
from random import randint
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics

data = pd.read_csv(r"G:\Учебка\Анализ данных\steam.csv", sep=',')

# Очистка данных
data = data.dropna() 
data = data.drop(columns=["appid","name","release_date","developer","publisher","platforms","categories","genres",
                          "steamspy_tags","average_playtime","owners",]) 
data = data[(data["median_playtime"] != 0)]
data = data[(data["price"] != 0)]

# Удаление выбросов
def remove_outliers_for_playtime(data):
    first_quartile = data["median_playtime"].describe()['25%']
    third_quartile = data["median_playtime"].describe()['75%']
            
    iqr = third_quartile - first_quartile

    data = data[(data["median_playtime"] > (first_quartile - 3 * iqr)) &
                (data["median_playtime"] < (third_quartile + 3 * iqr))]
    return data

def train_test_split(obj, train_percent, Y_column): # Поддерживает только один игрек
    msk = np.random.rand(len(obj)) < train_percent 
    # Делим на тестовую и обучающую
    train = obj[msk]
    test = obj[~msk]
    # Отделяем иксы игреки
    Xtrn = train.drop(columns=[Y_column])
    Ytrn = train[Y_column]
    Xtest = test.drop(columns=[Y_column])
    Ytest = test[Y_column] 
    return Xtrn, Ytrn, Xtest, Ytest
#______________________________
# Непосредственно лабортаторка
def predictRND(obj):
    for index, item in obj.iterrows():
        obj.loc[index,"class"] = randint(0,1)
    return obj

data = predictRND(data)
Xtrn, Ytrn, Xtest, Ytest = train_test_split(data, 0.8, 'class')
 
# Модельки
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

models = [DecisionTreeClassifier(criterion = 'entropy'), # Дерево решений
          RandomForestClassifier(n_estimators=100), # Случайный лес
          GradientBoostingClassifier(), # Гралиентный бустинг
	     ]

colors = ('b','y','g')

# создаем коллекции
Result = DataFrame() 
tmp = {}
i = 0

# Пошли по моделям
for model in models:
# получаем имя модели
   m = str(model)
   tmp['Модель'] = m[:m.index('(')]    
   # обучаем модель
   model.fit(Xtrn,Ytrn)
   Ypred = model.predict(Xtest)
   type(Ypred)
   # Метрки там вся фигня
   tmp['Оценка модели'] = classification_report(Ytest, Ypred)
   # ROC криывые
   fpr, tpr, threshold = metrics.roc_curve(Ytest, Ypred)
   roc_auc = metrics.auc(fpr, tpr)
   plt.plot(fpr, tpr, str(colors[i]), label = m[:m.index('(')] + 'AUC = %0.2f' % roc_auc)
   plt.legend(loc = 'lower right')
   plt.plot([0, 1], [0, 1],'r--')
   plt.xlim([0, 1])
   plt.ylim([0, 1])
   # записываем данные и итоговый DataFrame
   Result = Result.append([tmp])
   i += 1
Result.set_index('Модель', inplace=True)
plt.show()
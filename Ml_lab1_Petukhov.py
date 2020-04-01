# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:04:20 2020

@author: Petukhov Alexandr Sergeevich https://github.com/Tabaki59/ML-Labs
"""
# Задание 1
import pandas as pd
import numpy as np # надо для дат чтоб привести к месяцу
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Задание 1 импортируем файлы
transact = pd.read_csv(r"G:\Учебка\Машинка\resto-asap2018\casc-resto-small.csv", sep = ",")
print(transact)
print(transact.describe())

customers = pd.read_csv(r"G:\Учебка\Машинка\resto-asap2018\CASC_Constant.csv", sep = ",")
customers = customers.drop(columns=["ActivationDate","SubscribedEmail","SubscribedPush"])  # Сносим все лишнее, что потом не нужно будет притягивать
print(customers)
print(customers.describe())

# Задача 2 считаем хар-ки клиента 
transact.RKDate = pd.to_datetime(transact.RKDate) # Преобразуем дату из строки в ДАТУ
print(type(transact.RKDate[0]))
transact = transact.sort_values("RKDate") # Сортируем, пусть будет
print(transact.tail())

# Функция для расчета был ли визит (Если среди списка визитов клиента есть визит в указанном диапазоне ставим тру и едем дальше иначе оставляем false)
def get_visithappen(obj):
    result = False
    for index, item in obj.iterrows():
        if pd.to_datetime("2017-12-31") >= item['RKDate'] >= pd.to_datetime("2017-07-01"):
            result = True
    return result


# Функция для рассчета Recency
def get_recency(obj):
    #    Дата из задачи - дата последнего визита потом конвертим в дни, естественно смотрим на все визиты клиента чтоб найти последний
    df_with_correct_date = obj[obj['RKDate'] < pd.to_datetime("2017-07-01")]
    if not df_with_correct_date.empty:
        result = (pd.to_datetime("2017-07-01") - df_with_correct_date["RKDate"].max()) / np.timedelta64(1,'D')
    else:
        result = 0 
    return result
    
    
# Функция для рассчета Frequency    
def get_frequency(obj):
    # Считаем 2017 07 01  - дата первого визита 
    # Делим количество визитов на разницу в месяцах
    # return количество
    df_with_correct_date = obj[obj['RKDate'] < pd.to_datetime("2017-07-01")]
    if not df_with_correct_date.empty:
        month_count = (pd.to_datetime("2017-07-01") - df_with_correct_date["RKDate"].min()) / np.timedelta64(1,'M')
        visit_count =  len(df_with_correct_date.index) 
        result = visit_count / month_count
    else:
        result = 0
    return result 


# Функция для рассчета Monetary Value 
def get_monetary_value(obj):
    # средний чек клиента по всем покупкам до 2017-07-01
    # Бежим фором по прилетевшему дата фрейму, если дата меньше 2017 - 07 - 01 то суммируем чек, считаем чеки, потом делим одно на другое  
    count = 0 
    mon_value = 0
    for index, item in obj.iterrows():
        if item['RKDate'] < pd.to_datetime("2017-07-01"):
            count += 1
            mon_value += item['SummBasic']
    if count == 0:
        result = 0
        return result
    result = mon_value / count  
    count = 0 
    mon_value = 0
    return result     

    
transact_id = transact.groupby('CustomerID') # Группируем по id клиента

for index, item in transact.iterrows():  # Производим рассчет RFM переменных и Рассчет переменной Y 
    transact.loc[index,'Recency'] = get_recency(transact_id.get_group(item['CustomerID']))
    transact.loc[index,'Frequency'] = get_frequency(transact_id.get_group(item['CustomerID']))
    transact.loc[index,'MonetValue'] = get_monetary_value(transact_id.get_group(item['CustomerID']))
    transact.loc[index,"Visithappen"] = get_visithappen(transact_id.get_group(item['CustomerID']))  # Добавляем предсказываемую переменную был ли визит
    
transact = transact.drop_duplicates(subset=['CustomerID'])  # Сносим дубликаты по айдишникам чтоб каждый клиент и его хар-ки был представлен 1 раз
transact = transact.drop(columns=["RKDate","RegionName","BrandsNames", "DishCategoryName", "Quantity", "SummBasic", "SummAfterPointsUsage"])  # Сносим все лишнее, что теперь будет только мешаться

# Задание 3 притягиваем данные из второй таблицы
customers.rename(columns={'CustomerId': 'CustomerID'}, inplace=True)    # переименовываем столбец ибо изначально они не совпадают 
main_df = transact.merge(customers) # Лепим обе таблицы 
main_df = main_df.dropna() # Сносим пустые значения так как пол содержат NaN
main_df = main_df.replace({'Sex': {'Male': True,'Female': False}}) # Меняем полы на тип Boolean

# Задание 4 делим выборку на тестовую и обучающую
trg = main_df['Visithappen']  # Отделяем от выборки значение зависимой переменной
trn = main_df.drop(columns=['Visithappen'], axis=1)
Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size = 0.2) # Разбиваем выборку на тестовую и обучающую

# Задание 5 обучение модели
logistic = LogisticRegression(solver='lbfgs')
logistic.fit(Xtrn,Ytrn) # Обучаем модель
logistic.score(Xtrn, Ytrn) # На всякий случай производим оценку модели
print('Coefficient: \n', logistic.coef_)
print('Intercept: \n', logistic.intercept_)
print('R² Value: \n', logistic.score(Xtrn, Ytrn))

# Задание 6 предсказываем у
predict = pd.Series(logistic.predict(Xtest)) # Предсказываем значения у и конвертим в сериес
predict = predict.rename('Visithappen') # Проводим ренейм чтобы тестовый у и предсказанный хранились в одинаковом виде

# Задание 7 считаем метрики
report = classification_report(Ytest, predict) # Считаем метрики модели
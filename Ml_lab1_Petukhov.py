# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:04:20 2020

@author: User
"""
# Задание 1
import pandas as pd
import numpy as np # надо для дат чтоб привести к месяцу
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Задание 1 импортируем файлы
transact = pd.read_csv(r"G:\Учебка\Машинка\resto-asap2018\casc-resto-small.csv", sep = ",")
print(transact)
print(transact.describe())

customers = pd.read_csv(r"G:\Учебка\Машинка\resto-asap2018\CASC_Constant.csv", sep = ",")
customers = customers.drop(columns=["ActivationDate","SubscribedEmail","SubscribedPush"])  # Сносим все лишнее, что потом не нужно будет притягивать
print(customers)
print(customers.describe())

# Задача 2 
transact.RKDate = pd.to_datetime(transact.RKDate) # Преобразуем дату из строки в ДАТУ
print(type(transact.RKDate[0]))
transact = transact.sort_values("RKDate") # Сортируем, пусть будет
print(transact.tail())

# Функция для расчета был ли визит (Если среди списка визитов клиента есть визит в указанном диапазоне ставим тру и едем дальше)
def get_visithappen(obj):
    result = False
    for index, item in obj.iterrows():
        if pd.to_datetime("2017-12-31") >= item['RKDate'] >= pd.to_datetime("2017-07-01"):
            result = True
    return result


# Функция для рассчета Recency
def get_recency(obj):
    #    Вот вообще не понимаю что тут считать есть только идея дата из задачи - дата последнего визита но это просто догадка, впрочем рабочая
    df_with_correct_date = obj[obj['RKDate'] < pd.to_datetime("2017-07-01")]
    if not df_with_correct_date.empty:
        result = pd.to_datetime("2017-07-01") - df_with_correct_date["RKDate"].max()
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
    # Бежим фором по прилетевшему дата врейму, если дата меньше 2017 - 07 - 01 то суммируем чек, считаем чеки, потом делим одно на другое  
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


# Тестирую как играться с Groupby
#print(transact_unic_id.first())
#print(transact_unic_id.get_group(2728549)) 
#get_recency(transact_unic_id.get_group(2728549)) # Тест вызова функции (Кстати все ок работает) Плюс тесты работы с пандас
#print(get_monetary_value(transact_unic_id.get_group(2728549)))
#print(transact_unic_id.get_group(2728549)[transact_unic_id.get_group(2728549)['RKDate'] < pd.to_datetime("2017-07-01")]) 
    
transact_id = transact.groupby('CustomerID') # Группируем по id клиента

for index, item in transact.iterrows():  # Производим рассчет RFM переменных и Рассчет переменной Y 
    transact.loc[index,'Recency'] = get_recency(transact_id.get_group(item['CustomerID']))
    transact.loc[index,'Frequency'] = get_frequency(transact_id.get_group(item['CustomerID']))
    transact.loc[index,'MonetValue'] = get_monetary_value(transact_id.get_group(item['CustomerID']))
    transact.loc[index,"Visithappen"] = get_visithappen(transact_id.get_group(item['CustomerID']))  # Добавляем предсказываемую переменную был ли визит
    
transact = transact.drop_duplicates(subset=['CustomerID'])  # Сносим дубликаты по айдишникам чтоб каждый клиент и его хар-ки был представлен 1 раз
transact = transact.drop(columns=["RKDate","RegionName","BrandsNames", "DishCategoryName", "Quantity", "SummBasic", "SummAfterPointsUsage"])  # Сносим все лишнее, что теперь будет только мешаться

# Задание 3
customers.rename(columns={'CustomerId': 'CustomerID'}, inplace=True)    # переименовываем столбец ибо изначально они не совпадают 
main_df = transact.merge(customers) # Лепим обе таблицы 

# Задание 4
trg = main_df['Visithappen']  # Отделяем от выборки значение зависимой переменной
trn = main_df.drop(columns=['Visithappen'], axis=1)
Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size = 0.2) # Разбиваем выборку на тестовую и обучающую

# Задание 5 
logistic = LogisticRegression()
logistic.fit(Xtrn,Ytrn)
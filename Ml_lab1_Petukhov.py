# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:04:20 2020

@author: User
"""
# Задание 1
import pandas as pd

transact = pd.read_csv(r"G:\Учебка\Машинка\resto-asap2018\casc-resto-small.csv", sep = ",")
print(transact)
print(transact.describe())

customers = pd.read_csv(r"G:\Учебка\Машинка\resto-asap2018\CASC_Constant.csv", sep = ",")
print(customers)
print(customers.describe())

# Задача 2 
transact.RKDate = pd.to_datetime(transact.RKDate)
print(type(transact.RKDate[0]))
transact = transact.sort_values("RKDate")
print(transact.tail())

transact["visithappen"] = transact['RKDate'].between('2017-07-01','2017-12-31', inclusive=True)  
transact_unic_id = transact.groupby('CustomerID')

# Тестирую как играться с Groupby
print(transact_unic_id.first())
print(transact_unic_id.get_group(2728549)) 

def get_recency(obj, cust_id):
    #    Вот вообще не понимаю что тут считать есть только идея дата из задачи - дата первого визита но это просто потолок
    print(obj.RKDate)
    print('Да ты меня вызвал')
    
def get_frequency(obj, cust_id):
    # Считаем среднее количество визитов в месяц дату 2017 07 01 - дата первого визита смотрим на разницу в месяцах
    # Делим количество визитов на разницу в месяцах
    # return количество
    pass

def get_monetary_value(obj, cust_id):
    # средний чек клиента по всем покупкам до 2017-07-01
    pass

get_recency(transact_unic_id.get_group(2728549)) # Тест вызова функции (Кстати все ок работает)

#for index, item in transact.iterrows():
#    передаем в каждую функцию: transact_unic_id.get_group(item['CustomerID'])) и id клиента
    
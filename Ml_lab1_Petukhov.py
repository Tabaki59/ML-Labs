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

rkdate = transact.RKDate[(transact.RKDate >= pd.to_datetime('2017-07-01')) & (transact.RKDate <= pd.to_datetime('2017-12-31'))]
dates = transact.groupby(["CustomerID", rkdate]).count()['Restaurant']
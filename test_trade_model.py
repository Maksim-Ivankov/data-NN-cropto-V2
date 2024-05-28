

import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import time
from talib import abstract
import progressbar
import tensorflow as tf
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random


csv_file_path = 'test/'
# csv_file_path = 'big_data/ml_v1/5min/'
TP = 0.008
SL = 0.02

model = tf.keras.models.load_model('models/final/1000SATSUSDT.keras')

df = pd.read_csv(f'{csv_file_path}1000SATSUSDT.csv')

 # df_test = df.iloc[31500:-1]
df_test = df
# df_test = df.iloc[30300:-1]
# df_test = df.iloc[28980:-1]
trade_mas = []
scaler = StandardScaler()
X = scaler.fit_transform(df_test)
number_df_test = 0
predskasanie_celka = []
for row in X:
  y = model.predict(np.array([row], dtype=float))
  predskasanie_celka.append(y)
  if y[0][0]!=1 and y[0][1]!=1:
    if y[0][0]<0.1:long = 1
    elif y[0][0]>=0.9: long = 0
    else: long = 0
    if y[0][1]<0.1:short = 1
    elif y[0][1]>=0.9: short = 0
    else: short = 0
    trade_mas.append([number_df_test,long,short])
  number_df_test+=1
print(f'{len(df_test)} | {len(trade_mas)}')
# print(trade_mas)
# ниже торговля
flag_trade = 0
number_index_mass_trade = 0
deposit = 100 # депозит
plecho = 1 # плечо
depos_mass = {}
depos_mass_depo = []
depos_mass_nuber = []
long_result_plus = 0
long_result_minus = 0
short_result_plus = 0
short_result_minus = 0
long_trade = 0
short_trade = 0
number_trade = 0
for index, row in df_test.iterrows():
  if index>81:
    if flag_trade == 0:
      for res in trade_mas:
        # print(f'{index == (res[0]+31500)} | {index} | {res[0]+31500} |')
        if index == (res[0]):
        # if index == (res[0]+30300):
          if res[1] == 1 and res[2] == 1:
            # see_trade = 'long'
            # price_trade = row['open']
            # TP_price = price_trade*(1+TP) # тейк и стоп для лонга
            # SL_price = price_trade*(1-SL)
            # long_trade+=1
            # flag_trade = 1
            continue
          if res[1] == 1 and res[2] == 0:
            see_trade = 'long'
            price_trade = row['open']
            TP_price = price_trade*(1+TP) # тейк и стоп для лонга
            SL_price = price_trade*(1-SL)
            long_trade+=1
            flag_trade = 1
          elif res[1] == 0 and res[2] == 1:
            short_trade+=1
            see_trade = 'short'
            price_trade = row['open']
            TP_price = price_trade*(1-TP) # Тейк и стоп для шорта
            SL_price = price_trade*(1+SL)
            flag_trade = 1
          else: continue
    else:
      # print(f'Следим - {index}')
      if see_trade == 'long':
        if row['close']>TP_price or row['high']>TP_price:
            long_result_plus+=1 # Значит сработал тейк
            deposit = deposit + deposit*plecho*TP
            flag_trade = 0
            depos_mass_depo.append(deposit)
            depos_mass_nuber.append(number_trade)
            number_trade+=1
        elif row['close']<SL_price or row['low']<SL_price:
            long_result_minus+=1 # значит сработал стоп
            deposit = deposit - deposit*plecho*SL
            flag_trade = 0
            depos_mass_depo.append(deposit)
            depos_mass_nuber.append(number_trade)
            number_trade+=1
      else:
        if row['close']<TP_price or row['low']<TP_price:
            short_result_plus+=1 # Значит сработал тейк
            deposit = deposit + deposit*plecho*TP
            flag_trade = 0
            depos_mass_depo.append(deposit)
            depos_mass_nuber.append(number_trade)
            number_trade+=1
        elif row['close']>SL_price or row['high']>SL_price:
            short_result_minus+=1 # значит сработал стоп
            deposit = deposit - deposit*plecho*SL
            flag_trade = 0
            depos_mass_depo.append(deposit)
            depos_mass_nuber.append(number_trade)
            number_trade+=1
depos_mass['deposit'] = depos_mass_depo
depos_mass['number'] = depos_mass_nuber
depos_mass['long_result_plus'] = long_result_plus
depos_mass['long_result_minus'] = long_result_minus
depos_mass['short_result_plus'] = short_result_plus
depos_mass['short_result_minus'] = short_result_minus
depos_mass['long_trade'] = long_trade
depos_mass['short_trade'] = short_trade
depos_mass['predskasanie_celka'] = predskasanie_celka
print(f'Депозит = {deposit}')
print(f'Сделок {long_trade+short_trade} | В плюс {long_result_plus+short_result_plus} | В минус {long_result_minus+short_result_minus} | ')
# print(depos_mass)

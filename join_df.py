# Скрипт объединения датафреймов из папок по дням в единый фрейм за все месяцы по каждой монете отдельно.
# В итоге в паепке будут лежать csv с названиями монет, в которых будут объедененные данные за все дни торговли

import os
import pandas as pd
import time
from talib import abstract
import progressbar
from sklearn.preprocessing import StandardScaler
import numpy as np

path = 'big_data/big_data/'
csv_file_path = 'test_df/'
# csv_file_path = 'big_data/ml_v1/5min/'
# csv_file_path = 'big_data/ml_v1/5min/'

# # ШАГ 0
# # Проверка на последовательность вермени открытия
# print('Шаг 0. Проверка на последовательность вермени открытия')
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
# index_non = []
# for i in range(len(coin)):
#     coin[i] = coin[i][:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     df = pd.read_csv(f'{csv_file_path}{coin[i]}.csv')
#     index_non[:] = []
#     for index, row in df.iterrows():
#         if index>0:
#             if df['open_time'].iloc[index] - df['open_time'].iloc[index-1] != 60000:
#                 index_non.append(index)
#     print(f'{coin[i]} | {index_non}')
#     bar.update(i)
# print('Шаг 3. Выполнено')

# # # # ШАГ 0-1
# # Вставка недостающих данных
# print('Шаг 0. Вставка недостающих данных')
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
# index_non = []
# step_into = 129601
# for i in range(len(coin)):
#     coin[i] = coin[i][:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     # if coin[i] != ('ANTUSDT'):
#     df = pd.read_csv(f'{csv_file_path}{coin[i]}.csv')
#     df_once = df.iloc[0:step_into-1]
#     df_seredina = pd.read_csv(f'get_none_df/{step_into}/{coin[i]}.csv')
#     df_second = df.iloc[step_into:-1]
#     new_df = pd.concat([df_once,df_seredina,df_second], ignore_index=True, axis=0)
#     new_df.to_csv(f'{csv_file_path}{coin[i]}.csv', index=False)
#     # index_non[:] = []
#     # for index, row in df.iterrows():
#     #     if index>0:
#     #         if df['open_time'].iloc[index] - df['open_time'].iloc[index-1] != 60000:
#     #             index_non.append(index)
#     # print(f'{coin[i]} | {index_non}')
#     bar.update(i)
# print('Шаг 3. Выполнено')



# # ШАГ 1
# # получаем название коинов и чистим от багованых
# print('Шаг 1. Получаем название коинов и чистим от багованых. Приступаем')
# coin = next(os.walk(f'{path}1/'))[2] # получаем все названия файлов в папке 0 - монеты
# for i in range(len(coin)):
#     coin[i] = coin[i][:-4] # чистим названия от расширения .csv - оставляем только название монеты
# # coin.remove('GLMUSDT') # Чистим от монет, которые потерялись в процессе
# # coin.remove('OMUSDT')
# # coin.remove('VANRYUSDT')
# # coin.remove('ANTUSDT')
# # coin.remove('XMRUSDT')
# print('Шаг 1. Выполнено')

# # ШАГ 2
# # цикл ниже объединяет датафреймы с одинаковыми названиями файлов в один
# print('Шаг 2. Объединяем датафреймы с одинаковыми названиями файлов в один. Приступаем')
# arr = next(os.walk(f'{path}'))[1] # получаем все папки по пути
# arr = [eval(i) for i in arr] # преобразуем названия папок в числа
# bar = progressbar.ProgressBar(maxval=len(arr)).start() # прогресс бар в консоли
# step_2_num = 0
# for folder in sorted(arr): # итерируемся по отсортированному массиву с названием папок от 0 до N
#     bar.update(step_2_num)
#     step_2_num+=1
#     if folder !=0:
#         for coin_name in coin:
#             df = pd.read_csv(f'{path}{folder}/{coin_name}.csv')
#             df_res = pd.read_csv(f'{csv_file_path}{coin_name}.csv')
#             df_res = df_res._append(df.iloc[1:len(df)])
#             df_res.to_csv(f'{csv_file_path}{coin_name}.csv', index=False)
# print('Шаг 2. Выполнено')
            
# # ШАГ 3
# # цикл ниже удаляет ненужные столбцы
# print('Шаг 3. Удаляем ненужные столбцы. Приступаем')
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
# for i in range(len(coin)):
#     coin[i] = coin[i][:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     df = pd.read_csv(f'{csv_file_path}{coin[i]}.csv')
#     df_new = df.drop(['Unnamed: 0','close_time','open_time'], axis=1)
#     df_new.to_csv(f'{csv_file_path}{coin[i]}.csv', index=False)
#     bar.update(i)
# print('Шаг 3. Выполнено')

# # ШАГ 4/5
# # Суммируем предидущие N значений в один параметр
# print('Шаг 4/5. Суммируем предидущие N значений в один параметр')
# N = 10
# hall_mass = []
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
# count = 0
# open_sum_mas = []
# high_sum_mas = []
# low_sum_mas = []
# close_sum_mas = []
# volume_sum_mas = []
# for coin_name in coin:
#     coin_name = coin_name[:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     df = pd.read_csv(f'{csv_file_path}{coin_name}.csv')
#     bar.update(count)
#     count+=1
#     open_sum_mas[:] = []
#     high_sum_mas[:] = []
#     low_sum_mas[:] = []
#     close_sum_mas[:] = []
#     volume_sum_mas[:] = []
#     for index, row in df.iterrows():
#         if index>N:
#             open_sum = 0
#             high_sum = 0
#             low_sum = 0
#             close_sum = 0
#             volume_sum = 0
#             df_SUM = df.iloc[index-N:index]
#             for index_sum, row_sum in df_SUM.iterrows():
#                 open_sum += row_sum['open']
#                 high_sum += row_sum['high']
#                 low_sum += row_sum['low']
#                 close_sum += row_sum['close']
#                 volume_sum += row_sum['VOLUME']
#             open_sum_mas.append(open_sum)
#             high_sum_mas.append(high_sum)
#             low_sum_mas.append(low_sum)
#             close_sum_mas.append(close_sum)
#             volume_sum_mas.append(volume_sum)
        
#         else: 
#             open_sum_mas.append(0)
#             high_sum_mas.append(0)
#             low_sum_mas.append(0)
#             close_sum_mas.append(0)
#             volume_sum_mas.append(0)
#     df['open_sum'] = open_sum_mas
#     df['high_sum'] = high_sum_mas
#     df['low_sum'] = low_sum_mas
#     df['close_sum'] = close_sum_mas
#     df['volume_sum'] = volume_sum_mas
#     df.to_csv(f'{csv_file_path}{coin_name}.csv', index=False)  
# print('Шаг 5. Выполнено')  

# # ШАГ 4
# # Добавляем значения прижатия к коридору
# print('Шаг 4. Добавляем значения прижатия к коридору. Приступаем')
# hall_mass = []
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
# count = 0
# for coin_name in coin:
#     coin_name = coin_name[:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     df = pd.read_csv(f'{csv_file_path}{coin_name}.csv')
#     bar.update(count)
#     count+=1
#     a = abstract.BBANDS(df, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=0)
#     hall_mass[:] = []
#     for index, row in df.iterrows():
#         if index >4:
#             if (a['upperband'].iloc[index-1]-a['lowerband'].iloc[index-1]) == 0: # иногда индикатор показывает одинаковый верх и низ - ошибка, обнуляем ее занчение, чтобы потом вычистить
#                 hall_mass.append(0)
#             else:
#                 canal_procent = (df['close'].iloc[index-1]-a['lowerband'].iloc[index-1])/(a['upperband'].iloc[index-1]-a['lowerband'].iloc[index-1])
#                 hall_mass.append(round(canal_procent,5))
#         else:
#             hall_mass.append(0)
#     df['hall'] = hall_mass
#     df.to_csv(f'{csv_file_path}{coin_name}.csv', index=False)  
# print('Шаг 4. Выполнено')  

# # ШАГ 5
# # Добавляем индикаторы
# print('Шаг 5. Добавляем индикаторы. Приступаем')
# DEMA_mas  = []
# EMA_mas  = []
# HT_TRENDLINE_mas  = []
# KAMA_mas  = []
# MA_mas  = []
# MIDPOINT_mas  = []
# MIDPRICE_mas  = []
# SMA_mas  = []
# TEMA_mas  = []
# TRIMA_mas  = []
# WMA_mas  = []
# ADX_mas  = []
# ADXR_mas  = []
# APO_mas  = []
# AROON_mas_down = []
# AROON_mas_up = []
# BOP_mas  = []
# CCI_mas  = []
# CMO_mas  = []
# DX_mas  = []
# MACD_mas  = []
# MFI_mas  = []
# MINUS_DI_mas  = []
# MINUS_DM_mas  = []
# MOM_mas  = []
# PLUS_DI_mas  = []
# ROC_mas  = []
# CDLEVENINGSTAR_mas  = []
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# count_bar = 0
# for coin_name in coin:
#     coin_name = coin_name[:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     df = pd.read_csv(f'{csv_file_path}{coin_name}.csv')
#     df.rename(columns={'VOLUME':'volume'},inplace=True)
#     bar = progressbar.ProgressBar(maxval=len(df)).start() # прогресс бар в консоли
#     count_bar+=1
#     DEMA_mas[:] = []
#     EMA_mas[:] = []
#     HT_TRENDLINE_mas[:] = []
#     KAMA_mas[:] = []
#     MA_mas[:] = []
#     MIDPOINT_mas[:] = []
#     MIDPRICE_mas[:] = []
#     SMA_mas[:] = []
#     TEMA_mas[:] = []
#     TRIMA_mas[:] = []
#     WMA_mas[:] = []
#     ADX_mas[:] = []
#     ADXR_mas[:] = []
#     APO_mas[:] = []
#     AROON_mas_down[:] = []
#     AROON_mas_up[:] = []
#     BOP_mas[:] = []
#     CCI_mas[:] = []
#     CMO_mas[:] = []
#     DX_mas[:] = []
#     MACD_mas[:] = []
#     MFI_mas[:] = []
#     MINUS_DI_mas[:] = []
#     MINUS_DM_mas[:] = []
#     MOM_mas[:] = []
#     PLUS_DI_mas[:] = []
#     ROC_mas[:] = []
#     CDLEVENINGSTAR_mas[:] = []
#     DEMA = abstract.DEMA(df, timeperiod=25)
#     EMA = abstract.EMA(df, timeperiod=22)
#     HT_TRENDLINE = abstract.HT_TRENDLINE(df)
#     KAMA = abstract.KAMA(df, timeperiod=28)
#     MA = abstract.MA(df, timeperiod=18, matype=0)
#     MIDPOINT = abstract.MIDPOINT(df, timeperiod=8)
#     MIDPRICE = abstract.MIDPRICE(df, timeperiod=10)
#     SMA = abstract.SMA(df, timeperiod=24)
#     TEMA = abstract.TEMA(df, timeperiod=27)
#     TRIMA = abstract.TRIMA(df, timeperiod=21)
#     WMA = abstract.WMA(df, timeperiod=29)
#     ADX = abstract.ADX(df,13)
#     ADXR = abstract.ADXR(df,7)
#     APO = abstract.APO(df,14)
#     AROON = abstract.AROON(df,14)
#     BOP = abstract.BOP(df,14)
#     CCI = abstract.CCI(df,14)
#     CMO = abstract.CMO(df,14)
#     DX = abstract.DX(df,14)
#     MACD = abstract.MACD(df,12, 26, 9)
#     MFI = abstract.MFI(df,14)
#     MINUS_DI = abstract.MINUS_DI(df,14)
#     MINUS_DM = abstract.MINUS_DM(df,14)
#     MOM = abstract.MOM(df,14)
#     PLUS_DI = abstract.PLUS_DI(df,14)
#     ROC = abstract.ROC(df, 14)
#     CDLEVENINGSTAR = abstract.CDLEVENINGSTAR(df)
#     for index, row in df.iterrows():
#         bar.update(index)
#         if index >30:
#             DEMA_mas.append(round(DEMA[index-1],8))
#             EMA_mas.append(round(EMA[index-1],8))
#             HT_TRENDLINE_mas.append(round(HT_TRENDLINE[index-1],8))
#             KAMA_mas.append(round(KAMA[index-1],8))
#             MA_mas.append(round(MA[index-1],8))
#             MIDPOINT_mas.append(round(MIDPOINT[index-1],8))
#             MIDPRICE_mas.append(round(MIDPRICE[index-1],8))
#             SMA_mas.append(round(SMA[index-1],8))
#             TEMA_mas.append(round(TEMA[index-1],8))
#             TRIMA_mas.append(round(TRIMA[index-1],8))
#             WMA_mas.append(round(WMA[index-1],8))
#             ADX_mas.append(round(ADX[index-1],8))
#             ADXR_mas.append(round(ADXR[index-1],8))
#             APO_mas.append(round(APO[index-1],8))
#             AROON_mas_down.append(round(AROON['aroondown'][index-1],8))
#             AROON_mas_up.append(round(AROON['aroonup'][index-1],8))
#             BOP_mas.append(round(BOP[index-1],8))
#             CCI_mas.append(round(CCI[index-1],8))
#             CMO_mas.append(round(CMO[index-1],8))
#             DX_mas.append(round(DX[index-1],8))
#             MACD_mas.append(round(MACD['macdsignal'][index-1],8))
#             MFI_mas.append(round(MFI[index-1],8))
#             MINUS_DI_mas.append(round(MINUS_DI[index-1],8))
#             MINUS_DM_mas.append(round(MINUS_DM[index-1],8))
#             MOM_mas.append(round(MOM[index-1],8))
#             PLUS_DI_mas.append(round(PLUS_DI[index-1],8))
#             ROC_mas.append(round(ROC[index-1],8))
#             CDLEVENINGSTAR_mas.append(round(CDLEVENINGSTAR[index-1],8))
#         else:
#             DEMA_mas.append(0)
#             EMA_mas.append(0)
#             HT_TRENDLINE_mas.append(0)
#             KAMA_mas.append(0)
#             MA_mas.append(0)
#             MIDPOINT_mas.append(0)
#             MIDPRICE_mas.append(0)
#             SMA_mas.append(0)
#             TEMA_mas.append(0)
#             TRIMA_mas.append(0)
#             WMA_mas.append(0)
#             ADX_mas.append(0)
#             ADXR_mas.append(0)
#             APO_mas.append(0)
#             AROON_mas_down.append(0)
#             AROON_mas_up.append(0)
#             BOP_mas.append(0)
#             CCI_mas.append(0)
#             CMO_mas.append(0)
#             DX_mas.append(0)
#             MACD_mas.append(0)
#             MFI_mas.append(0)
#             MINUS_DI_mas.append(0)
#             MINUS_DM_mas.append(0)
#             MOM_mas.append(0)
#             PLUS_DI_mas.append(0)
#             ROC_mas.append(0)
#             CDLEVENINGSTAR_mas.append(0)
#     df['DEMA'] = DEMA_mas
#     df['EMA'] = EMA_mas
#     df['HT_TRENDLINE'] = HT_TRENDLINE_mas
#     df['KAMA'] = KAMA_mas
#     df['MA'] = MA_mas
#     df['MIDPOINT'] = MIDPOINT_mas
#     df['MIDPRICE'] = MIDPRICE_mas
#     df['SMA'] = SMA_mas
#     df['TEMA'] = TEMA_mas
#     df['TRIMA'] = TRIMA_mas
#     df['WMA'] = WMA_mas
#     df['ADX'] = ADX_mas
#     df['ADXR'] = ADXR_mas
#     df['APO'] = APO_mas
#     df['AROON_down'] = AROON_mas_down
#     df['AROON_up'] = AROON_mas_up
#     df['BOP'] = BOP_mas
#     df['CCI'] = CCI_mas
#     df['CMO'] = CMO_mas
#     df['DX'] = DX_mas
#     df['MACD'] = MACD_mas
#     df['MFI'] = MFI_mas
#     df['MINUS_DI'] = MINUS_DI_mas
#     df['MINUS_DM'] = MINUS_DM_mas
#     df['MOM'] = MOM_mas
#     df['PLUS_DI'] = PLUS_DI_mas
#     df['ROC'] = ROC_mas
#     df['CDLEVENINGSTAR'] = CDLEVENINGSTAR_mas
#     df.to_csv(f'{csv_file_path}{coin_name}.csv', index=False)  
#     print(f'{count_bar}/{len(coin)} | {coin_name} добавлен')
# print('Шаг 5. Выполнено')  

# ШАГ 6
# Добавляем значения прижатия к коридору
print('Шаг 6. Чистим от первых 31 пустых значений. Приступаем')
hall_mass = []
coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
count_step_6 = 0
# print(coin)
for coin_name in coin:
    coin_name = coin_name[:-4] # чистим названия от расширения .csv - оставляем только название монеты
    df = pd.read_csv(f'{csv_file_path}{coin_name}.csv')
    bar.update(count_step_6)
    count_step_6+=1
    df = df.drop (df.index[8000:-1]) 
    df.to_csv(f'{csv_file_path}{coin_name}.csv', index=False) 
print('Шаг 6. Сделано') 


# 7 шаг - это торговля на табличных данных - trade.py


# ШАГ 8
# # обрезаем последние 16 элементов в дф
# print('Шаг 7. Обрезаем последние 16 элементов в дф. Приступаем')
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
# count_step_6 = 0
# for coin_name in coin:
#     coin_name = coin_name[:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     df = pd.read_csv(f'{csv_file_path}{coin_name}.csv')
#     bar.update(count_step_6)
#     count_step_6+=1
#     df = df.drop (index=[len(df)-1,len(df)-2,len(df)-3,len(df)-4,len(df)-5,len(df)-6,len(df)-7,len(df)-8,len(df)-9,len(df)-10,len(df)-11,len(df)-12,len(df)-13,len(df)-14,len(df)-15,len(df)-16]) 
#     df.to_csv(f'{csv_file_path}{coin_name}.csv', index=False) 
# print('Шаг 8. Сделано') 


# # # ШАГ 9
# # # цикл ниже удаляет ненужные столбцы
# print('Шаг 9. Удаляем ненужные столбцы. Приступаем')
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
# for i in range(len(coin)):
#     coin[i] = coin[i][:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     df = pd.read_csv(f'{csv_file_path}{coin[i]}.csv')
#     df_new = df.drop(['ADX','ADXR','APO','BOP','CCI','CMO','DX','MACD','MFI','MINUS_DI','MINUS_DM','MOM','PLUS_DI','ROC','long','short'], axis=1)
#     df_new.to_csv(f'{csv_file_path}{coin[i]}.csv', index=False)
#     bar.update(i)
# print('Шаг 9. Выполнено')


# # ШАГ 10
# # Нормализация данных - пока не работает
# print('Шаг 9. Нормализация данных. Приступаем')
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
# for i in range(len(coin)):
#     coin[i] = coin[i][:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     df = pd.read_csv(f'{csv_file_path}{coin[i]}.csv')
    
#     X = df[df.columns[:-1]].values
#     y = df[df.columns[-1]].values

#     images_flat = tf.contrib.layers.flatten(X)

#     data = np.hstack((images_flat,np.reshape(y,(-1,1))))
#     DF = pd.DataFrame(data) 

#     DF.to_csv(f'{csv_file_path}{coin[i]}.csv', index=False)
#     bar.update(i)
#     time.sleep(20000)
# print('Шаг 9. Выполнено')


# ШАГ 6
# Добавляем значения прижатия к коридору
# print('Шаг 6. Чистим от первых 31 пустых значений. Приступаем')
# hall_mass = []
# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
# bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
# count_step_6 = 0
# for coin_name in coin:
#     coin_name = coin_name[:-4] # чистим названия от расширения .csv - оставляем только название монеты
#     df = pd.read_csv(f'{csv_file_path}{coin_name}.csv')
#     bar.update(count_step_6)
#     count_step_6+=1
#     df = df.drop (index=[ 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) 
#     df.to_csv(f'{csv_file_path}{coin_name}.csv', index=False) 
# print('Шаг 6. Сделано') 





















































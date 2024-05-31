
import os
import pandas as pd
import time
from talib import abstract
import progressbar

path = 'big_data/5min/'
csv_file_path = 'big_data/ml_v1/1min/'

TP = 0.003
SL = 0.003
TIME_TRADE = 5 # 12 пятиминуток стои в сделке - 60 минут. За это время должны словить тейк или стоп, если не словим - сделка 0

coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты
bar = progressbar.ProgressBar(maxval=len(coin)).start() # прогресс бар в консоли
count_step_6 = 0
mass_long_result = []
# mass_short_result = []
for coin_name in coin:
    coin_name = coin_name[:-4] # чистим названия от расширения .csv - оставляем только название монеты
    df = pd.read_csv(f'{csv_file_path}{coin_name}.csv')
    bar.update(count_step_6)
    count_step_6+=1
    # mass_short_result[:] = []
    mass_long_result[:] = []
    # условия для выбора направления сделки:
    for index, row in df.iterrows():
        if index < (len(df)-14): # ограничиваем фрейм сверху, так как не сможем проверить - удачная сделка или нет. Потом нужно будет обрезать рфейм сверху на 16 значений
            # Сначала рассматриваем только лонги
            price_trade = row['open']
            df_trade = df.iloc[index:index+TIME_TRADE]
            TP_price = price_trade*(1+TP) # тейк и стоп для лонга
            SL_price = price_trade*(1-SL)
            for index_now, row_now in df_trade.iterrows(): # условие для лонга
                if row_now['close']>TP_price or row_now['high']>TP_price:
                    long_result=1 # Значит сработал тейк
                    break
                elif row_now['close']<SL_price or row_now['low']<SL_price:
                    long_result=0 # значит сработал стоп
                    break
                else:
                    long_result=0# стоим в сделке слишком долго, не влезли в TIME_TRAADE
            mass_long_result.append(long_result)
            # TP_price = price_trade*(1-TP) # Тейк и стоп для шорта
            # SL_price = price_trade*(1+SL)
            # for index_now, row_now in df_trade.iterrows(): # условие для шорта
            #     if row_now['close']<TP_price or row_now['low']<TP_price:
            #         short_result=1 # Значит сработал тейк
            #         break
            #     elif row_now['close']>SL_price or row_now['high']>SL_price:
            #         short_result=0 # значит сработал стоп
            #         break
            #     else:
            #         short_result=0 # стоим в сделке слишком долго, не влезли в TIME_TRAADE
            # mass_short_result.append(short_result)
        else:
            # mass_short_result.append(0)
            mass_long_result.append(0)
    # df['long'] = mass_short_result
    df['long'] = mass_long_result
    df.to_csv(f'{csv_file_path}{coin_name}.csv', index=False)  




















































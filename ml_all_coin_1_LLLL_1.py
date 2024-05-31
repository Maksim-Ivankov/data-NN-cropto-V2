
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import time
from talib import abstract
import progressbar
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import logging
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
tf.autograph.set_verbosity(3)


csv_file_path = 'big_data/ml_v1/1min/'


# coin = next(os.walk(f'{csv_file_path}/'))[2] # получаем все названия файлов в папке 0 - монеты

# Сама модель
def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs,activat_1,activat_2,activat_3,potery):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(dropout_prob, activation=activat_1, input_shape=(4,)),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(num_nodes, activation=activat_2),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])


  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
  history = nn_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size,shuffle=True, validation_split=0.2
    # X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks = [model_checkpoint]
  )

  return nn_model, history

# получаем в нампи разделенные x и y и массив целиком, передав датафрейм
def scale_dataset(dataframe):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1:]].values
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  # data = np.hstack((X, y))
  data = np.hstack((X, np.reshape(y, (-1, 1))))
  return data, X, y

# отрисовка и сохранение графика
def plot_history(history,title,model,i,val_loss,coin_up,TP,SL,y_train,y_valid):
  print_log(coin_up,i)
  print_log(title,i)
  data_trade = print_trade(model,coin_up,i,TP,SL,y_train,y_valid)
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
  ax1.plot(history.history['loss'], label='Потери')
  ax1.plot(history.history['val_loss'], label='Значение потери')
  ax1.set_xlabel('Эпохи')
  ax1.set_ylabel('Бинарная кроссэнтропия')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='Точность')
  ax2.plot(history.history['val_accuracy'], label='Значение точности')
  ax2.set_xlabel('Эпохи')
  ax2.set_ylabel('точность')
  ax2.grid(True)
  

  ax3.plot(data_trade['deposit'], label='Депозит')
  # ax3.plot(data_trade['number'], label='Номер сделки')
  ax3.set_xlabel('Номер сделки')
  ax3.set_ylabel('Депозит')
  ax3.grid(True)

  ax1.set_title(title)
  ax2.set_title(f'{round(data_trade['tochnost_istina'],2)} | {round(data_trade['tochnost_istina_long'],2)}')
  ax3.set_title(f'Л+ {data_trade['long_result_plus']}|Л- {data_trade['long_result_minus']}||Лонгов {data_trade['long_trade']}')
  # plt.figtext(0, 0, data_trade['predskasanie_celka'], fontsize=6)
  plt.savefig(f'graph/{i}.png')
  # plt.show()

# торговля по текущей модели
def print_trade(model,coin_up,step,TP,SL,y_train,y_valid):
  print('Закончили обучать модель')
  hall_mass = []
  trade_mas = []
  df_coin = pd.read_csv(f'test_df/{coin_up}.csv')
  df_coin_test_15 = pd.read_csv(f'test_15/{coin_up}.csv')
  df_coin.rename(columns={'VOLUME':'volume'},inplace=True)
  print('Получаем предсказания из реального ДФ, шаг {step} | ')
  # Предсказания по реальным данным
  trade_mas = []
  scaler = StandardScaler()
  X = scaler.fit_transform(df_coin_test_15)
  number_df_test = 0
  predskasanie_celka = []
  pred_real = model.predict(X,verbose = 0)
  Long_lvl = max(pred_real[0])*0.9
  for lol in pred_real:
    if lol[0]>Long_lvl: long = 1
    else: long = 0
    trade_mas.append([number_df_test,long])
    number_df_test+=1

  # Считаем истиную точность
  TIME_TRADE = 30
  mass_long_result = []
  mass_long_result[:] = []
  for index, row in df_coin.iterrows():
    if index < (len(df_coin)-30): # ограничиваем фрейм сверху, так как не сможем проверить - удачная сделка или нет. Потом нужно будет обрезать рфейм сверху на 16 значений
        # Сначала рассматриваем только лонги
        price_trade = row['open']
        df_trade = df_coin.iloc[index:index+TIME_TRADE]
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
    else:
        mass_long_result.append(0)

  count_true = 0
  count_true_odin = 0
  count_true_all = 0
  for i in range(len(df_coin)):
     if mass_long_result[i] == trade_mas[i][1]:
        count_true+=1
     if mass_long_result[i] == 1 and trade_mas[i][1]==1:
        count_true_odin+=1
     if mass_long_result[i] == 1:
        count_true_all+=1
  tochnost_istina = (count_true/len(df_coin))*100
  tochnost_istina_long = (count_true_odin/count_true_all)*100
  print_log(f'ТОЧНОСТЬ ИСТИНА : {tochnost_istina} %',step)
  print_log(f'ТОЧНОСТЬ ЛОНГОВ : {tochnost_istina_long} %',step)

  print_log(str(predskasanie_celka),step)
  print_log('------------------------',step)
  print_log(str(trade_mas),step)
  print_log('------------------------',step)
  print_log(f'Сигналов всего {len(trade_mas)}',step)
  print_log(f'Минимальное занчане предсказания : {min(pred_real)}',step)
  print_log(f'Максимальное значение предсказания : {max(pred_real)}',step)
  print_log('------------------------',step)

  # ниже торговля
  print('Закончили собирать предсказания, начинаем торговлю')
  flag_trade = 0
  number_index_mass_trade = 0
  deposit = 100 # депозит
  plecho = 1 # плечо
  depos_mass = {}
  depos_mass_depo = []
  depos_mass_nuber = []
  long_result_plus = 0
  long_result_minus = 0
  long_trade = 0
  bar2 = progressbar.ProgressBar(maxval=len(df_coin)).start() # прогресс бар в консоли
  number_trade = 0
  for index, row in df_coin.iterrows():
    bar2.update(index)
    if flag_trade == 0:
      for res in trade_mas:
        # print(f'{index == (res[0]+31500)} | {index} | {res[0]+31500} |')
        if index == (res[0]):
          if res[1] == 1:
            price_trade = row['close']
            TP_price = price_trade*(1+TP) # тейк и стоп для лонга
            SL_price = price_trade*(1-SL)
            long_trade+=1
            flag_trade = 1
            
          else: continue
    else:
      # print(f'Следим - {index}')
      if row['close']>TP_price or row['high']>TP_price:
          long_result_plus+=1 # Значит сработал тейк
          deposit = deposit + deposit*plecho*TP
          flag_trade = 0
          depos_mass_depo.append(deposit)
          depos_mass_nuber.append(number_trade)
          number_trade+=1
          print_log(f'Сделка {long_trade} | {coin_up} | Депо {deposit} | Шаг {index} ',step)
      elif row['close']<SL_price or row['low']<SL_price:
          long_result_minus+=1 # значит сработал стоп
          deposit = deposit - deposit*plecho*SL
          flag_trade = 0
          depos_mass_depo.append(deposit)
          depos_mass_nuber.append(number_trade)
          number_trade+=1
    if deposit<60:
       break
  depos_mass['deposit'] = depos_mass_depo
  depos_mass['number'] = depos_mass_nuber
  depos_mass['long_result_plus'] = long_result_plus
  depos_mass['long_result_minus'] = long_result_minus
  depos_mass['long_trade'] = long_trade
  depos_mass['predskasanie_celka'] = predskasanie_celka
  depos_mass['tochnost_istina'] = tochnost_istina
  depos_mass['tochnost_istina_long'] = tochnost_istina_long
  print_log(f'------------------------',step)
  print_log(f'Результат - {deposit}',step)
  print_log(f'Количество сделок - {len(depos_mass_depo)}',step)
  print_log(f'Сделок в плюс : {long_result_plus}',step)
  print_log(f'Сделок в минус : {long_result_minus}',step)

  return(depos_mass)

# принтуем логи в файл
def print_log(msg,i):
    path =f'log/{i}.txt'
    f = open(path,'a',encoding='utf-8')
    f.write('\n'+msg)
    f.close()

epochs=100
for i in range(50):


  # coin_up = random.choice(['1000SATSUSDT', '1INCHUSDT', 'AAVEUSDT', 'ACEUSDT', 'ACHUSDT', 'ADAUSDT', 'AEVOUSDT', 'AGIXUSDT', 'AGLDUSDT', 'AIUSDT', 'ALGOUSDT', 'ALICEUSDT', 'ALPHAUSDT', 'ALTUSDT'])
  # coin_up = random.choice(['AMBUSDT', 'ANKRUSDT', 'APEUSDT', 'API3USDT', 'APTUSDT', 'ARBUSDT', 'ARKMUSDT', 'ARKUSDT', 'ARPAUSDT', 'ARUSDT', 'ASTRUSDT', 'ATAUSDT', 'ATOMUSDT', 'AUCTIONUSDT'])
  # coin_up = random.choice(['AUDIOUSDT', 'AVAXUSDT', 'AXLUSDT', 'AXSUSDT', 'BADGERUSDT', 'BAKEUSDT', 'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BCHUSDT', 'BEAMXUSDT', 'BELUSDT', 'BICOUSDT', 'BLURUSDT'])
  # coin_up = random.choice(['BLZUSDT', 'BNBUSDT', 'BNTUSDT', 'BNXUSDT', 'BOMEUSDT', 'BONDUSDT', 'BTCUSDT', 'C98USDT', 'CAKEUSDT', 'CELOUSDT', 'CELRUSDT', 'CFXUSDT', 'CHRUSDT', 'CHZUSDT'])
  # coin_up = random.choice(['CKBUSDT', 'COMBOUSDT', 'COMPUSDT', 'COTIUSDT', 'CRVUSDT', 'CTKUSDT', 'CTSIUSDT', 'CVCUSDT', 'CVXUSDT', 'CYBERUSDT', 'DARUSDT', 'DASHUSDT', 'DENTUSDT', 'DGBUSDT'])
  # coin_up = random.choice(['DOGEUSDT', 'DOTUSDT', 'DUSKUSDT', 'DYDXUSDT', 'DYMUSDT', 'EDUUSDT', 'EGLDUSDT', 'ENAUSDT', 'ENJUSDT', 'ENSUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHFIUSDT', 'ETHUSDT'])
  # coin_up = random.choice(['FETUSDT', 'FILUSDT', 'FLMUSDT', 'FLOWUSDT', 'FRONTUSDT', 'FTMUSDT', 'FTTUSDT', 'FXSUSDT', 'GALAUSDT', 'GALUSDT', 'GASUSDT', 'GLMRUSDT', 'GLMUSDT', 'GMTUSDT'])
  # coin_up = random.choice(['GMXUSDT', 'GRTUSDT', 'GTCUSDT', 'HBARUSDT', 'HFTUSDT', 'HIFIUSDT', 'HIGHUSDT', 'HOOKUSDT', 'HOTUSDT', 'ICPUSDT', 'ICXUSDT', 'IDEXUSDT', 'IDUSDT', 'ILVUSDT'])
  # coin_up = random.choice(['IMXUSDT', 'INJUSDT', 'IOSTUSDT', 'IOTAUSDT', 'IOTXUSDT', 'JASMYUSDT', 'JOEUSDT', 'JTOUSDT', 'JUPUSDT', 'KAVAUSDT', 'KEYUSDT', 'KLAYUSDT', 'KNCUSDT', 'KSMUSDT'])
  coin_up = random.choice(['LDOUSDT', 'LEVERUSDT', 'LINAUSDT', 'LINKUSDT', 'LITUSDT', 'LOOMUSDT', 'LPTUSDT', 'LQTYUSDT', 'LRCUSDT', 'LSKUSDT', 'LTCUSDT', 'MAGICUSDT', 'MANAUSDT'])
  # coin_up = random.choice(['MAGICUSDT'])
  
  # 'MASKUSDT', 'MATICUSDT', 'MAVUSDT', 'MBLUSDT', 'MDTUSDT', 'MEMEUSDT', 'METISUSDT', 'MINAUSDT', 'MKRUSDT', 'MOVRUSDT', 'MTLUSDT', 'NEARUSDT', 'NEOUSDT', 'NFPUSDT', 
  # 'NKNUSDT', 'NMRUSDT', 'NTRNUSDT', 'OCEANUSDT', 'OGNUSDT', 'OMGUSDT', 'OMNIUSDT', 'OMUSDT', 'ONEUSDT', 'ONGUSDT', 'ONTUSDT', 'OPUSDT', 'ORDIUSDT', 'OXTUSDT', 
  # 'PENDLEUSDT', 'PEOPLEUSDT', 'PERPUSDT', 'PHBUSDT', 'PIXELUSDT', 'POLYXUSDT', 'PORTALUSDT', 'POWRUSDT', 'PYTHUSDT', 'QNTUSDT', 'QTUMUSDT', 'RADUSDT', 'RAYUSDT', 
  # 'RDNTUSDT', 'REEFUSDT', 'RENUSDT', 'RIFUSDT', 'RLCUSDT', 'RNDRUSDT', 'RONINUSDT', 'ROSEUSDT', 'RSRUSDT', 'RUNEUSDT', 'RVNUSDT', 'SAGAUSDT', 'SANDUSDT', 'SCUSDT', 
  # 'SEIUSDT', 'SFPUSDT', 'SKLUSDT', 'SLPUSDT', 'SNTUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT', 'SSVUSDT', 'STEEMUSDT', 'STGUSDT', 'STMXUSDT', 'STORJUSDT', 'STPTUSDT', 
  # 'STRAXUSDT', 'STRKUSDT', 'STXUSDT', 'SUIUSDT', 'SUPERUSDT', 'SUSHIUSDT', 'SXPUSDT', 'TAOUSDT', 'THETAUSDT', 'TIAUSDT', 'TLMUSDT', 'TNSRUSDT', 'TRBUSDT', 'TRUUSDT', 
  # 'TRXUSDT', 'TUSDT', 'TWTUSDT', 'UMAUSDT', 'UNFIUSDT', 'UNIUSDT', 'USDCUSDT', 'USTCUSDT', 'VANRYUSDT', 'VETUSDT', 'WAVESUSDT', 'WAXPUSDT', 'WIFUSDT', 'WLDUSDT', 
  # 'WOOUSDT', 'WUSDT', 'XAIUSDT', 'XEMUSDT', 'XLMUSDT', 'XRPUSDT', 'XTZUSDT', 'XVGUSDT', 'XVSUSDT', 'YFIUSDT', 'YGGUSDT', 'ZECUSDT', 'ZENUSDT', 'ZILUSDT', 'ZRXUSDT.csv'

  try:
    df = pd.read_csv(f'{csv_file_path}{coin_up}.csv')
    X = df[df.columns[:-1]].values
    y = df[df.columns[-1]].values
    scaler = StandardScaler() # нормализация данных
    X = scaler.fit_transform(X)# нормализация данных
    over = RandomOverSampler() # эти две строки уровняют количество 0 и 1, так лучше будет работать нейронка
    X,y = over.fit_resample(X,y)
    data = np.hstack((X,np.reshape(y, (-1,1)))) # для графиков
    transformed_df = pd.DataFrame(data, columns=df.columns) # для графиков
    train_pct_index = int(0.7 * len(X))
    X_train, X_valid = X[:train_pct_index], X[train_pct_index:]
    y_train, y_valid = y[:train_pct_index], y[train_pct_index:]
    # X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.2, random_state=0) # 80% пойдет в X_train, 20% - в X_temp
    # X_valid, X_test, y_valid, y_test = train_test_split(X_temp,y_temp, test_size=0.5, random_state=0) # из 20% сверху дробим 50 на 50 на новалидационные и тестовые
    # train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
    # train_data, X_train, y_train = scale_dataset(train)
    # valid_data, X_valid, y_valid = scale_dataset(valid)
    # test_data, X_test, y_test = scale_dataset(test)
    TP = random.choice([0.003,0.005,0.007,0.009])
    SL = random.choice([0.003,0.005,0.007,0.009,0.02,0.04,0.06,0.08,0.1])
    num_nodes = random.choice([2,5,10,15,20,30,50,100])
    dropout_prob = random.choice([2,5,10,15,20,30,50,100])
    lr = random.choice([0.01, 0.005, 0.001])
    batch_size = random.choice([256])
    activat_1 = random.choice(['elu','exponential','gelu','hard_sigmoid','hard_silu','hard_swish','leaky_relu','linear','log_softmax','mish','relu','selu','sigmoid','silu','softmax','softplus','softsign','swish','tanh'])
    activat_2 = random.choice(['elu','exponential','gelu','hard_sigmoid','hard_silu','hard_swish','leaky_relu','linear','log_softmax','mish','relu','selu','sigmoid','silu','softmax','softplus','softsign','swish','tanh'])
    activat_3 = random.choice(['elu','exponential','gelu','hard_sigmoid','hard_silu','hard_swish','leaky_relu','linear','log_softmax','mish','relu','selu','sigmoid','silu','softmax','softplus','softsign','swish','tanh'])
    potery = random.choice(['binary_crossentropy','MAE','MAPE','MSE','MSLE','binary_focal_crossentropy','categorical_crossentropy','categorical_focal_crossentropy','cosine_similarity','dice','hinge','huber','kld','mae','mape','mse','msle','poisson','squared_hinge','tversky'])
    title_graph = f"{num_nodes}|{dropout_prob}|{lr}|{batch_size}|{activat_1}\n|{activat_2}|{activat_3}|{potery}|{TP}|{SL}"
    print(f'Начали обучать модель | Монета {coin_up} | {title_graph}')
    model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs,activat_1,activat_2,activat_3,potery)
    # val_loss = model.evaluate(X_valid, y_valid)[0]
    val_loss = 1
    plot_history(history,title_graph,model,i,val_loss,coin_up,TP,SL,y_train,y_valid)
    model.save(f'models/{i}.keras')
  except Exception as e:
    print(f'Ошибка - {e}')

    # matrix_train = confusion_matrix(y_train, pred_train)
    # matrix_test = confusion_matrix(y_valid, pred_test)

    # print(matrix_train)
    # print(matrix_test)






























































































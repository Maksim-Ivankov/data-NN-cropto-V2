
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

import logging
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
tf.autograph.set_verbosity(3)


csv_file_path = 'big_data/ml_v1/1min/'


# coin_up = random.choice(['1000SATSUSDT', '1INCHUSDT', 'AAVEUSDT', 'ACEUSDT', 'ACHUSDT', 'ADAUSDT', 'AEVOUSDT', 'AGIXUSDT', 'AGLDUSDT', 'AIUSDT', 'ALGOUSDT', 'ALICEUSDT', 'ALPHAUSDT', 'ALTUSDT.csv'])
# coin_up = random.choice(['AMBUSDT', 'ANKRUSDT', 'APEUSDT', 'API3USDT', 'APTUSDT', 'ARBUSDT', 'ARKMUSDT', 'ARKUSDT', 'ARPAUSDT', 'ARUSDT', 'ASTRUSDT', 'ATAUSDT', 'ATOMUSDT', 'AUCTIONUSDT.csv'])
# coin_up = random.choice(['AUDIOUSDT', 'AVAXUSDT', 'AXLUSDT', 'AXSUSDT', 'BADGERUSDT', 'BAKEUSDT', 'BALUSDT', 'BANDUSDT', 'BATUSDT', 'BCHUSDT', 'BEAMXUSDT', 'BELUSDT', 'BICOUSDT', 'BLURUSDT.csv'])
# coin_up = random.choice(['BLZUSDT', 'BNBUSDT', 'BNTUSDT', 'BNXUSDT', 'BOMEUSDT', 'BONDUSDT', 'BTCUSDT', 'C98USDT', 'CAKEUSDT', 'CELOUSDT', 'CELRUSDT', 'CFXUSDT', 'CHRUSDT', 'CHZUSDT.csv'])
# coin_up = random.choice(['CKBUSDT', 'COMBOUSDT', 'COMPUSDT', 'COTIUSDT', 'CRVUSDT', 'CTKUSDT', 'CTSIUSDT', 'CVCUSDT', 'CVXUSDT', 'CYBERUSDT', 'DARUSDT', 'DASHUSDT', 'DENTUSDT', 'DGBUSDT.csv'])
# coin_up = random.choice(['DOGEUSDT', 'DOTUSDT', 'DUSKUSDT', 'DYDXUSDT', 'DYMUSDT', 'EDUUSDT', 'EGLDUSDT', 'ENAUSDT', 'ENJUSDT', 'ENSUSDT', 'EOSUSDT', 'ETCUSDT', 'ETHFIUSDT', 'ETHUSDT.csv'])
# coin_up = random.choice(['FETUSDT', 'FILUSDT', 'FLMUSDT', 'FLOWUSDT', 'FRONTUSDT', 'FTMUSDT', 'FTTUSDT', 'FXSUSDT', 'GALAUSDT', 'GALUSDT', 'GASUSDT', 'GLMRUSDT', 'GLMUSDT', 'GMTUSDT.csv'])
# coin_up = random.choice(['GMXUSDT', 'GRTUSDT', 'GTCUSDT', 'HBARUSDT', 'HFTUSDT', 'HIFIUSDT', 'HIGHUSDT', 'HOOKUSDT', 'HOTUSDT', 'ICPUSDT', 'ICXUSDT', 'IDEXUSDT', 'IDUSDT', 'ILVUSDT.csv'])
coin_up = random.choice(['IMXUSDT', 'INJUSDT', 'IOSTUSDT', 'IOTAUSDT', 'IOTXUSDT', 'JASMYUSDT', 'JOEUSDT', 'JTOUSDT', 'JUPUSDT', 'KAVAUSDT', 'KEYUSDT', 'KLAYUSDT', 'KNCUSDT', 'KSMUSDT.csv'])


df = pd.read_csv(f'{csv_file_path}{coin_up[0]}.csv')

X = df[df.columns[:-1]].values
y = df[df.columns[-1]].values
scaler = StandardScaler() # нормализация данных
X = scaler.fit_transform(X)# нормализация данных
over = RandomOverSampler() # эти две строки уровняют количество 0 и 1, так лучше будет работать нейронка
X,y = over.fit_resample(X,y)
data = np.hstack((X,np.reshape(y, (-1,1)))) # для графиков
transformed_df = pd.DataFrame(data, columns=df.columns) # для графиков

train_pct_index = int(0.7 * len(X))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]
























































import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing

SEQ_LEN = 60  # # какую длину предшествующей последовательности нужно собрать для RNN
FUTURE_PERIOD_PREDICT = 3  # как далеко в будущем мы пытаемся заглянуть?
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10  # сколько раз проходят через наши данные
BATCH_SIZE = 64  # сколько пакетов? Попробуйте использовать меньший пакет, если у вас возникают ошибки OOM (нехватка памяти).
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):  # если будущая цена выше текущей, это означает покупку или 1
        return 1
    else:  # в противном случае... это 0!
        return 0


def preprocess_df(df):
    df = df.drop("future", 1)  # мне это больше не нужно.

    for col in df.columns:  # пройдитесь по всем колонкам
        if col != "target":  # нормализуйте все ... за исключением самой цели!
            df[col] = df[col].pct_change()  # изменение pct "нормализует" различные валюты (каждая криптовалюта имеет значительно различающиеся значения, нас действительно больше интересует движение другой монеты).
            df.dropna(inplace=True)  # удалите сетевой накопитель, созданный с помощью pct_change
            df[col] = preprocessing.scale(df[col].values)  # шкала от 0 до 1.

    df.dropna(inplace=True)  # шкала от 0 до 1.


    sequential_data = []  # это список, который будет содержать следующие последовательности
    prev_days = deque(maxlen=SEQ_LEN)  # Это и будут наши реальные последовательности. Они созданы с помощью deque, который сохраняет максимальную длину, удаляя старые значения по мере ввода новых

    for i in df.values:  # выполните итерацию по значениям
        prev_days.append([n for n in i[:-1]])  #хранить все, кроме целевого
        if len(prev_days) == SEQ_LEN:  # убедитесь, что у нас есть 60 последовательностей!
            sequential_data.append([np.array(prev_days), i[-1]])  # добавьте этих плохих парней!

    random.shuffle(sequential_data)  # перемешайте для пущей убедительности.

    buys = []  # список, в котором будут храниться наши последовательности покупок и целевые показатели
    sells = []  # список, в котором будут храниться наши последовательности продаж и целевые показатели

    for seq, target in sequential_data:  # выполните итерацию по последовательным данным
        if target == 0:  # если это "не покупать"
            sells.append([seq, target])  # добавить в список продавцов
        elif target == 1:  # в противном случае, если цель равна 1...
            buys.append([seq, target])  # это покупка!

    random.shuffle(buys)  # перемешайте покупки
    random.shuffle(sells)  # перемешайте ракушки!

    lower = min(len(buys), len(sells))  # какая длина короче?

    buys = buys[:lower]  # убедитесь, что оба списка имеют минимальную длину.
    sells = sells[:lower]  # убедитесь, что оба списка имеют минимальную длину.

    sequential_data = buys+sells  # сложите их вместе
    random.shuffle(sequential_data)  # еще одна перетасовка, чтобы модель не перепуталась сначала с одним классом, а затем с другим.

    X = []
    y = []

    for seq, target in sequential_data:  # просматриваем наши новые последовательные данные
        X.append(seq)  # X - это последовательности
        y.append(target)  #y - это цели/метки (покупает или продает/не покупает).

    return np.array(X), y  # возвращаем X и y...и превращаем X в числовой массив!


main_df = pd.DataFrame() # начинайте с нуля

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # 4 соотношения, которые мы хотим рассмотреть
for ratio in ratios:  # начать итерацию

    ratio = ratio.split('.csv')[0]  # отделите бегущую строку от имени файла
    print(ratio)
    dataset = f'training_datas/{ratio}.csv'  # получите полный путь к файлу.
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # чтение в определенном файле

    # переименуйте volume и close, чтобы включить тикер, чтобы мы могли определить, какое значение имеет close/volume.:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # установите время в качестве индекса, чтобы мы могли присоединиться к ним в это общее время
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # игнорируйте другие столбцы, кроме цены и объема

    if len(main_df)==0:  # если фрейм данных пуст
        main_df = df  #тогда это просто текущий df
    else:  # в противном случае присоедините эти данные к основным
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  # если в данных есть пробелы, используйте ранее известные значения
main_df.dropna(inplace=True)
#print(main_df.head())  # как у нас получилось??

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

main_df.dropna(inplace=True)

## здесь выделите некоторый фрагмент будущих данных из основного main_df.
times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # уникальное имя файла, которое будет содержать эпоху и код проверки для этой эпохи
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # сохраняет только лучшие из них

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Оценочная модель
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))
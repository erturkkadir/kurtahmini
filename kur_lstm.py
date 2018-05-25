import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

windows_size = 70       # Pencere boyutu
output_size = 2         # sonraki kac deger tahmin edilecek
epochs = 700            # islemi tekrar sayisi

features = 3


def get_data(y):
    # lstm datayi (rows, timesteps, features) formatinda ister
    # rows : ornek sayimiz,
    # timesteps : zaman adimlari
    # features : ogrenme verimizin sutun sayisi
    # training verisi (rows, 1, features)
    # output verisi (rows, 1)

    train_size = int(len(y)*0.7)                # Verinin %70 ini egitim icin kalaninin da test icin ayiracagiz

    train = y[0:train_size]
    test = y[train_size:len(y)]
    print(train.shape)
    print(test.shape)

    ############## TRAIN DATA ####################
    train_x = []
    train_y = []
    for i in range(0, train_size - features - output_size):
        tmp_x = y[i:(i+features)]
        tmp_y = y[(i+features):(i+features+output_size)]
        train_x.append(np.reshape(tmp_x, (1, features)))
        train_y.append(tmp_y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    print("train_x shape : ", train_x.shape)
    print("train_y shape : ", train_y.shape)

    ########### TEST DATA ########################
    test_x = []
    test_y = []
    last = len(y) - features - output_size
    for i in range(train_size, last):
        tmp_x = y[i:(i + features)]
        tmp_y = y[(i + features):(i + features + output_size)]
        test_x.append(np.reshape(tmp_x, (1, features)))
        test_y.append(tmp_y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)
    print("train_x shape : ", train_x.shape)
    print("train_y shape : ", train_y.shape)

    ######## Tahmin edilecek data #######################
    data_x = []
    tmp_x = y[-features:len(y)]
    data_x.append(np.reshape(tmp_x, (1, features)))
    data_x = np.array(data_x)

    print("data_x  : ", data_x)
    print("data_x shape : ", data_x.shape)

    return train_x, train_y, test_x, test_y, data_x


raw_data = pd.read_csv('./datasets/ue128.csv', header=None, names=["i", "t", "y"])
t = np.array(raw_data.t.values)
y = np.array(raw_data.y.values)

min = y.min()
max = y.max()

y = np.interp(y, (min, max), (-1, +1))

x_train, y_train, x_test, y_test, data_x = get_data(y)

model = Sequential()
model.add(LSTM(5, input_shape=(1, features), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(7))
model.add(Dense(output_size))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, verbose=0)

score = model.evaluate(x_test, y_test)
print("%2s: %.2f%%" % (model.metrics_names[1], score[1]*100))
model.summary()

data_y = model.predict(data_x)

result = np.interp(data_y, (-1, +1), (min, max))

print("Gelecekteki Degerler (output_size) :", result)

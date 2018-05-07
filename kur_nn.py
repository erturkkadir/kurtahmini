import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler


def get_data(y):
    train_size = int(len(y)*0.7)                # Verinin %70 ini egitim icin kalaninin da test icin ayiracagiz, 180

    ############## TRAIN DATA ####################
    train_x = []
    train_y = []
    for i in range(0, train_size):
        train_x.append(y[i:i+window_size])
        train_y.append((y[i+window_size:i+window_size+output_size]))

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    ########### TEST DATA ########################
    test_x = []
    test_y = []
    last = len(y)-output_size-window_size
    for i in range(train_size, last):
        test_x.append(y[i:i+window_size])
        test_y.append(y[i+window_size:i+window_size+output_size])

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    ######## Tahmin edilecek data #######################
    data_x = [y[-window_size:len(y)]]
    data_x = np.array(data_x)

    return train_x, train_y, test_x, test_y, data_x


window_size = 6     # Onceki 6 degeri kullan
output_size = 4     # sonraki 4 degeri tahmin etmeye calis
batch_size = 64     # her bir batinda kac satirda islem yapayim

epochs = 500        # islemi 500 kez tekrar et

raw_data = pd.read_csv('./datasets/ue128.csv', header=None, names=["i", "t", "y"])
t = np.array(raw_data.t.values)
y = np.array(raw_data.y.values)

min = y.min()
max = y.max()

y = np.interp(y, (min, max), (-1, +1))

x_train, y_train, x_test, y_test, data_x = get_data(y)
print("x_train  Shape = ", x_train.shape)
print("y_train  Shape = ", y_train.shape)
print("x_test  Shape = ", x_test.shape)
print("y_test  Shape = ", y_test.shape)
print("pred_x  Shape = ", data_x.shape)

model = Sequential()
model.add(Dense(32, input_dim=window_size, activation='relu'))
model.add(Dense(64))
model.add(Dense(output_size))

print("Shape Train x : ", x_train.shape)
print("Shape Train y : ", y_train.shape)

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size)

score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("Score : ", score)
model.summary()

data_y = model.predict(data_x)
print("Gelecekteki degerler : ", data_y)

print("Gelecekteki Degerler (output_size) :", np.interp(data_y, (-1, +1), (min, max)))

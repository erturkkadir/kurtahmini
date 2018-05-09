import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense


windows_size = 70       # Onceki 6 degeri kullan
output_size = 2         # sonraki 4 degeri tahmin etmeye calis
batch_size = 8          # her bir batinda kac satirda islem yapayim
epochs = 700            # islemi tekrar sayisi


def get_data(y):

    train_size = int(len(y)*0.7)                # Verinin %70 ini egitim icin kalaninin da test icin ayiracagiz, 180

    ############## TRAIN DATA ####################
    train_x = []
    train_y = []
    for i in range(0, train_size):
        train_x.append(y[i:i + windows_size])
        train_y.append((y[i + windows_size:i + windows_size + output_size]))

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    ########### TEST DATA ########################
    test_x = []
    test_y = []
    last = len(y) - output_size - windows_size
    for i in range(train_size, last):
        test_x.append(y[i:i + windows_size])
        test_y.append(y[i + windows_size:i + windows_size + output_size])

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    ######## Tahmin edilecek data #######################
    data_x = [y[-windows_size:len(y)]]
    data_x = np.array(data_x)

    return train_x, train_y, test_x, test_y, data_x


raw_data = pd.read_csv('./datasets/ue128.csv', header=None, names=["i", "t", "y"])
t = np.array(raw_data.t.values)
y = np.array(raw_data.y.values)

min = y.min()
max = y.max()

y = np.interp(y, (min, max), (-1, +1))

x_train, y_train, x_test, y_test, data_x = get_data(y)

model = Sequential()
model.add(Dense(32, input_dim=windows_size, activation='relu'))
model.add(Dense(64))
model.add(Dense(output_size))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size)

score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("%2s: %.2f%%" % (model.metrics_names[1], score[1]*100))
model.summary()

data_y = model.predict(data_x)

result = np.interp(data_y, (-1, +1), (min, max))

print("Gelecekteki Degerler (output_size) :", result)

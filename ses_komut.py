import keras
from ses_funcs import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

feature_dim_1 = 20
feature_dim_2 = 40

channel = 1
epochs = 75
num_classes = 4
batch_size = 20

x_train, x_test, y_train, y_test = get_train_test(feature_dim_1, feature_dim_2)

x_train = x_train.reshape(x_train.shape[0], feature_dim_1, feature_dim_2, channel)
x_test = x_test.reshape(x_test.shape[0], feature_dim_1, feature_dim_2, channel)


def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2,2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(Conv2D(48, kernel_size=(2,2), activation='relu'))
    model.add(Conv2D(120, kernel_size=(2,2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def predict(filepath, model):
    sample = wav2mfcc(filepath, feature_dim_1, feature_dim_2)
    sample_reshape = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    pred = model.predict(sample_reshape)
    print("Prediction : ", pred)
    label = get_labels()[0][np.argmax(pred)]
    return label


model = get_model()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
model.save('model.tf')

pred = predict('datasets/sesler/yukari/yukari1.wav', model=model)
print(pred)
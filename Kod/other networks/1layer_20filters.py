import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import *

data = pd.read_csv('sudoku_train.csv')

# reshape sudoku data into multidimensional arrays
X = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9, 1)) for flatten_grid in data.quizzes])
y = np.array([np.reshape([int(d) for d in flatten_grid], (81, 1)) for flatten_grid in data.solutions])

X_train = X[:900000]
y_train = y[:900000]

X_test = X[900000:]
y_test = y[900000:]

# subtract 1 from all output values in order to obtain [0,8] output interval
y_train = y_train - 1
y_test = y_test - 1

model = Sequential()

model.add(Conv2D(20, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))

model.add(Flatten())
model.add(Dense(729)) # 81*9
model.add(Reshape((81, 9))) # 9 possible cell values for each of the 81 squares on the board
model.add(Activation('softmax'))

model.summary()

adam = keras.optimizers.Adam(lr=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# callbacks used for training optimisation
callback1 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
callback2 = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=5,verbose=1)

model.fit(X_train, y_train, batch_size=320, epochs=100, validation_data=(X_test,y_test), callbacks=[callback1,callback2])

model.save("model_1l_20f")
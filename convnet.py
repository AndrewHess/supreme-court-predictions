from util.data import split_data
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Activation, Dropout, Flatten, Dense
import numpy as np

# Get the data.
train_x, train_y, test_x, test_y = split_data()

# Create the model.
model = Sequential()

model.add(Dense(32, input_shape=(149, 1)))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(8))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# Compile and test the model.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

results = model.fit(
    train_x, train_y,
    epochs= 100,
    batch_size = 100,
    validation_data = (test_x, test_y)
#    validation_data = (train_x, train_y)
)

print("Test-Accuracy:", np.mean(results.history["val_acc"]))

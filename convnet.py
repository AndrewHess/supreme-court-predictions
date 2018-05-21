from util.data import split_data
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Activation, Dropout, Flatten, Dense
import numpy as np

# Get the data.
train_x, train_y, test_x, test_y = split_data()
'''
print('------------------ test x ------------------')
print(test_x)
print('------------------ test y ------------------')
print(test_y)
for item in test_y:
    print(test_y)

print('train_y shape:', train_y.shape)
for i in range(train_y.shape[0]):
    print(train_y[i][0], end=' ')
'''


# Create the model.
model = Sequential()
'''
model.add(Conv1D(16, 4, input_shape=(11, 1)))
model.add(Activation('relu'))
#model.add(MaxPooling1D())

model.add(Conv1D(16, 4))
model.add(Activation('relu'))
#model.add(MaxPooling1D())

model.add(Conv1D(32, 4))
model.add(Activation('relu'))
#model.add(MaxPooling1D())
'''

'''
model.add(Dense(16, input_shape=(10, 1)))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Conv1D(128, 4))
model.add(Dense(256))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''

'''
model.add(Dense(16, input_shape=(10, 1)))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10))

model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''

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

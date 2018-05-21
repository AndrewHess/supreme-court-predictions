from util.data import split_data
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Activation, Dropout, Flatten, Dense
import numpy as np

# Get the data.
train_x, train_y, test_x, test_y = split_data()
result = [0, 0] # First is correct, second is incorrect prediction.

for i in range(len(test_x)):
    case = test_x[i]
    one = case[0][0]
    two = case[1][0]
    prediction = 0

    #if two > 280 or (two > 50 and (one > 75 and one < 280)):
    #    prediction = 0

    result[prediction ^ test_y[i][0]] += 1

print('Accuracy:', (result[0] / len(test_x)))

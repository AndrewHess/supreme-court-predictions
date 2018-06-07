from util.data import split_data
import numpy as np

# Get the data.
train_x, train_y, test_x, test_y = split_data()
result = [0, 0] # First is correct, second is incorrect prediction.

for i in range(len(test_x)):
    prediction = 0
    result[prediction ^ test_y[i][0]] += 1

print('Accuracy:', (result[0] / len(test_x)))

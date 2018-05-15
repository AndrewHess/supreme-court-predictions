import os
import csv
import numpy as np
import random

def split_data():
    train_folder = 'data.nosync/train/'
    test_folder  = 'data.nosync/test/'

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    train_x, train_y = get_file_contents(train_folder)
    test_x, test_y = get_file_contents(test_folder)

    # Shuffle the data but keep the labels or each point, so that training and
    # testing is not all points from one class and then all points from the
    # other class.
    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)

    # Convert the sets the numpy arrays.
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # Reshape the arrays and add 1 as the last dimension for x for Keras.
    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], 1))
    train_y = train_y.reshape((train_y.shape[0], 1))
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], 1))
    test_y = test_y.reshape((test_y.shape[0], 1))

    return (train_x, train_y, test_x, test_y)


def get_file_contents(base_path):
    contents = []
    labels = []
    folder_ind = 0

    for folder in sorted(os.listdir(base_path)):
        path = os.path.join(base_path, folder)
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), 'r') as in_file:
                # Read the csv file as a numpy array.
                reader = csv.reader(in_file, delimiter=',')

                for row in reader:
                    # Only include the row if it has a value for each field.
                    try:
                        contents.append(np.array(row).astype('float32'))
                        labels.append(folder_ind)
                        print('shape:', np.array(row).astype('float32').shape())
                    except:
                        # Do nothing
                        pass
        folder_ind += 1

    return (contents, labels)


def shuffle(data, labels):
    assert(len(data) == len(labels))

    # Make (datum, label) tuples so that the correct label is used.
    tuples = []

    for i in range(len(data)):
        tuples.append((data[i], labels[i]))

    random.shuffle(tuples)

    # Undo the (datum, label) tuples
    new_data, new_labels = [], []

    for i in range(len(tuples)):
        new_data.append(tuples[i][0])
        new_labels.append(tuples[i][1])

    return new_data, new_labels

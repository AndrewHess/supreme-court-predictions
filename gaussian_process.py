from util.data import split_data
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


# Get the data.
print('getting data')
train_x, train_y, test_x, test_y = split_data(dim=2)

# Create the model.
print('setting up model')
clf = GaussianProcessClassifier(1.0 * RBF(1.0))
print('fitting model')
clf.fit(train_x, train_y)

# Test the model.
print('testing model')
print('Train Accuracy:', clf.score(train_x, train_y))
print('Test Accuracy:', clf.score(test_x, test_y))

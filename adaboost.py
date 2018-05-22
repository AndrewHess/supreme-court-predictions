from util.data import split_data
from sklearn.ensemble import AdaBoostClassifier

# Get the data.
train_x, train_y, test_x, test_y = split_data(dim=2)

# Create the model.
clf = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
clf.fit(train_x, train_y)

# Test the model.
print('Train Accuracy:', clf.score(train_x, train_y))
print('Test Accuracy:', clf.score(test_x, test_y))

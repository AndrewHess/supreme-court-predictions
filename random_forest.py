from util.data import split_data
from sklearn.ensemble import RandomForestClassifier

# Get the data.
train_x, train_y, test_x, test_y = split_data(dim=2)

# Create the model.
clf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_features='auto', min_samples_split=15, n_jobs=-1)
clf.fit(train_x, train_y)

# Test the model.
print('Train Accuracy:', clf.score(train_x, train_y))
print('Test Accuracy:', clf.score(test_x, test_y))

import random

test_size = 0.2  # The fraction of data used for testing

random.seed(3)

# Open the files
data = open('../raw_data/full_feature_cases.csv', 'r')
result = open('../raw_data/full_feature_outcomes.csv', 'r')

train0 = open('../data.nosync/train/0/data.csv', 'w+')
train1 = open('../data.nosync/train/1/data.csv', 'w+')
test0 = open('../data.nosync/test/0/data.csv', 'w+')
test1 = open('../data.nosync/test/1/data.csv', 'w+')

for outcome in result:
    outcome = outcome.strip()  # Remove whitespace
    case = data.readline()

    # Ignore outcomes that are not 0 or 1
    if len(outcome) == 0 or (outcome != '0' and outcome != '1'):
        continue

    # Randomly add this case to either test or train.
    filename = None

    if (random.random() > test_size):
        if (outcome == '0'):
            filename = train0
        elif (outcome == '1'):
            filename = train1
        else:
            print('unknown outcome:', outcome)
    else:
        if (outcome == '0'):
            filename = test0
        elif (outcome == '1'):
            filename = test1
        else:
            print('unknown outcome:', outcome)

    # Now that we have the correct file, we just write it.
    filename.write(case)

# Clean up the files
result.close()
data.close()
train0.close()
train1.close()
test0.close()
test1.close()

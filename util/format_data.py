import random

test_size = 0.2  # The fraction of data used for testing
counters = [0] * 4  # File number counters for train and test, 1 and 2

random.seed(3)

# Open the files
data = open('../data.nosync/cases.csv', 'r')
result = open('../data.nosync/outcomes.csv', 'r')

train0 = open('../data.nosync/train/0/data.csv', 'w+')
train1 = open('../data.nosync/train/1/data.csv', 'w+')
test0 = open('../data.nosync/test/0/data.csv', 'w+')
test1 = open('../data.nosync/test/1/data.csv', 'w+')

for outcome in result:
    outcome = outcome.strip()  # Remove whitespace
    case = data.readline()

    # Ignore outcomes that are not 1 or 2
    if (len(outcome) == 0 or int(outcome) > 1):
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

    '''
    # Randomly add this case to either test or train.
    filename = '../data.nosync/test/'
    train = False

    if (random.random() > test_size):
        filename = '../data.nosync/train/'
        train = True

    # Add the case to the correct folder.
    filename += outcome + '/'
    filename += str(counters[2 * train + int(outcome) - 1])
    filename += '.csv'

    # Update the counter
    counters[2 * train + int(outcome) - 1] += 1

    # Put this case in the appropriate file.
    with open(filename, 'w+') as new_file:
        new_file.write(data.readline())
    '''


# Clean up the files
result.close()
data.close()
train0.close()
train1.close()
test0.close()
test1.close()

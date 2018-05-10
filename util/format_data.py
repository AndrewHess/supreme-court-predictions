import random

test_size = 0.2  # The fraction of data used for testing
counters = [0] * 4  # File number counters for train and test, 1 and 2

random.seed(3)

# Open the files
data = open('../data/numeric.csv', 'r')
result = open('../data/decision.csv', 'r')

for outcome in result:
    outcome = outcome.strip()  # Remove whitespace

    # Ignore outcomes that are not 1 or 2
    if (len(outcome) == 0 or int(outcome) > 2):
        continue

    # Randomly add this case to either test or train.
    filename = '../data/test/'
    train = False

    if (random.random() > test_size):
        filename = '../data/train/'
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


# Clean up the files
result.close()
data.close()
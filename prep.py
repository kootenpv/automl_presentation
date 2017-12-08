import pandas as pd

TITANIC_FILE_PATH = 'data/titanic_data.csv'

data = pd.read_csv(TITANIC_FILE_PATH)
y = data['Survived']

# prep for mlbox, nasty order
TRAIN_FILE = 'data/titanic_train.csv'
TEST_FILE = 'data/titanic_test.csv'
data[::2].to_csv(TRAIN_FILE)
data['Survived'] = None
data[1::2].to_csv(TEST_FILE)

data.drop('PassengerId', 1, inplace=True)
data.drop('Survived', 1, inplace=True)
X = data
del data

X_train, y_train = X[0::2], y[0::2]
X_test, y_test = X[1::2], y[1::2]

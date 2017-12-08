from tpot import TPOTClassifier

from prep import X_train, y_train, X_test, y_test

from xtoy import Sparsify

s = Sparsify()
X_train = s.fit_transform(X_train, y_train).toarray()
X_test = s.transform(X_test).toarray()

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_exports/tpot_titanic_pipeline.py')

# from prep import X, y

# training_features, training_target = X[::2], y[::2]
# testing_features, testing_target = X[1::2], y[1::2]

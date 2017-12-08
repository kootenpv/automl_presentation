# note to self: python3.7
from prep import X_train, y_train, X_test, y_test
import autosklearn.classification
import numpy as np
from xtoy import Sparsify

s = Sparsify()
X_train = s.fit_transform(X_train, y_train).toarray()
X_test = s.transform(X_test).toarray()


automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30)
automl.fit(X_train, np.array(y_train))

preds = automl.predict(X_test)
print("crossval autosklearn", (preds == y_test).mean())

# In [9]: (preds==y_test).mean()
# Out[9]: 0.80000000000000004

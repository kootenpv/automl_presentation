from xtoy import Toy
from prep import X_train, y_train, X_test, y_test

toy = Toy()
toy.fit(X_train, y_train)

y_preds = toy.predict(X_test)

print("crossval xtoy", (y_preds == y_test).mean())

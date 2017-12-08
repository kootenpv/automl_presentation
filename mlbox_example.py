from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *
from prep import TRAIN_FILE, TEST_FILE, y_test

paths = [TRAIN_FILE, TEST_FILE]
target_name = "Survived"
data = Reader(sep=",").train_test_split(paths, target_name)
data = Drift_thresholder().fit_transform(data)

opt = Optimiser(n_folds=3)
space = {

    'ne__numerical_strategy': {"search": "choice",
                               "space": [0]},
    'ce__strategy': {"search": "choice",
                     "space": ["label_encoding", "random_projection", "entity_embedding"]},
    'fs__threshold': {"search": "uniform",
                      "space": [0.01, 0.3]},
    'est__max_depth': {"search": "choice",
                       "space": [3, 4, 5, 6, 7]}

}

best = opt.optimise(space, data, 15)

prd = Predictor()
prd.fit_predict(best, data)


preds = pd.read_csv("save/Survived_predictions.csv")
np.mean(y_test.ravel() == preds["Survived_predicted"].ravel())

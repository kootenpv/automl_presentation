from xtoy import Sparsify
from prep import X

s = Sparsify(regex_patterns={"name_title": ", ([^.]+)"})

s.fit_transform(X)

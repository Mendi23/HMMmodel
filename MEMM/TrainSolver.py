from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import sys
from MEMM.MEMMTaggers import MemmTagger
from time import time
import numpy as np

if __name__ == '__main__':
    starttime = time()
    feature_vec_file, model_file = sys.argv[1:]
    x_train, y_train = load_svmlight_file(feature_vec_file, zero_based=True, dtype=np.int64)
    tagger = MemmTagger(model = LogisticRegression())
    tagger.fitModel(x_train, y_train)
    tagger.saveModelTofile(model_file)
    print(time()-starttime)
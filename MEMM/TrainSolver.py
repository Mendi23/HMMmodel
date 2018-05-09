from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import sys
from MEMM.MEMMTaggers import MemmTagger
import numpy as np
from scripts_t.measuretime import measure


@measure
def main(vec_file, modelf):
    x_train, y_train = load_svmlight_file(vec_file, zero_based=True, dtype=np.int64)
    tagger = MemmTagger(model=LogisticRegression())
    tagger.fitModel(x_train, y_train)
    tagger.saveModelTofile(modelf)


if __name__ == '__main__':
    feature_vec_file, model_file = sys.argv[1:3]
    main(feature_vec_file, model_file)

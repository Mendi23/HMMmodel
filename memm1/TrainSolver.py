# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import sys
from utils.MEMM_Taggers import MemmTagger
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

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import sys
from MEMM.MEMMTaggers import MemmTagger

if __name__ == '__main__':
    feature_vec_file, model_file = sys.argv[1:]
    x_train, y_train = load_svmlight_file(feature_vec_file)
    #print(x_train)
    lr = LogisticRegression(class_weight="balanced", random_state  = 200, solver = "liblinear")
    tagger = MemmTagger(model = lr)
    tagger.fitModel(x_train, y_train)
    tagger.saveModelTofile(model_file)
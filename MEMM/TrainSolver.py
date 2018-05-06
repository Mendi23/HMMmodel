from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import sys

if __name__ == '__main__':
    feature_vec_file, model_file = sys.argv[1:]
    x_train, y_train = load_svmlight_file(feature_vec_file)
    lr = LogisticRegression(class_weight="balanced", random_state  = 200, solver = "liblinear")
    lr.fit(x_train, y_train)

    with open(model_file, "w") as output:
        output.write(lr.get_params())
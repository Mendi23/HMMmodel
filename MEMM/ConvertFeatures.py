import sys
import pandas
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


def train_features(features_file):
    tags_set, features_set = set(), set()
    with open(features_file) as fIn:
        for line in fIn:
            splitted = line.split()
            tag, features = splitted[0], splitted[1:]
            tags_set.add(tag)
            for feature in features:
                features_set.add(feature)
    tags_encoder = LabelEncoder()
    features_encoder = LabelEncoder()
    tags_encoder.fit(list(tags_set))
    features_encoder.fit(list(features_set))
    return tags_encoder, features_encoder


def transform_features (tags_encoder, features_encoder, features_file, feature_vecs_file, feature_map_file):
    with open(features_file) as fIn, open(feature_vecs_file, "w") as fOut:
        for line in fIn:
            splitted = line.split()
            tag, features = splitted[0], splitted[1:]
            fOut.write("{tagnum} {features}\n".format(
                tagnum=tags_encoder.transform([tag])[0],
                features=' '.join((f"{fnum}:1" for fnum in features_encoder.transform(features)))
            ))

    with open(feature_map_file, "w") as fOut:
        for name in features_encoder.classes_:
            fOut.write(f"{name} {features_encoder.transform([name])[0]}\n")


if __name__ == '__main__':

    # features_file, feature_vecs_file, feature_map_file = sys.argv[1:]
    features_file, feature_vecs_file, feature_map_file = "features_file", "feature_vecs_file", "feature_map_file"

    tags_encoder, features_encoder = train_features(features_file)

    transform_features(tags_encoder, features_encoder, features_file, feature_vecs_file, feature_map_file)

import sys


def transform_features (fname, fvec_out, fmap_out):
    tags_dict = {}
    features_dict = {}
    t_i, f_i = 1, 1
    with open(fname) as fIn, \
            open(fvec_out, "w") as fOut, \
            open(fmap_out, "w") as mapOut:
        for line in fIn:
            splitted = line.split()
            tag, features = splitted[0], splitted[1:]

            if tags_dict.setdefault(tag, t_i) == t_i:
                t_i += 1

            for feature in features:
                if features_dict.setdefault(feature, f_i) == f_i:
                    mapOut.write(f"{feature} {f_i}\n")
                    f_i += 1

            sorted_features = sorted((features_dict[feature] for feature in features))
            fOut.write("{tag} {features}\n".format(
                tag=tags_dict[tag],
                features=' '.join((
                    f"{num}:1" for num in sorted_features
                ))
            ))

        mapOut.write("-----Tags Mapping-----\n")
        for tag, tag_num in tags_dict.items():
            mapOut.write(f"{tag} {tag_num}\n")


if __name__ == '__main__':
    features_file, feature_vecs_file, feature_map_file = sys.argv[1:]

    transform_features(features_file, feature_vecs_file, feature_map_file)

import sys
from collections import deque
from time import time

def transform_features_2(fname, fvec_out, fmap_out):
    tags_dict = {}
    features_dict = {}
    t,f = 1, 1
    with open(fname) as fIn, \
        open(fvec_out, "w") as fOut, \
        open(fmap_out, "w") as mapOut:
        for line in fIn:
            splitted = line.split()
            tag, features = splitted[0], splitted[1:]

            if tags_dict.setdefault(tag, t) == t:
                t += 1
            out_line = deque([f"{tags_dict[tag]}"])

            for feature in features:
                if features_dict.setdefault(feature, f) == f:
                    mapOut.write(f"{feature} {f}\n")
                    f += 1
                out_line.append(f"{features_dict[feature]}:1")

            fOut.write(f"{' '.join(out_line)}\n")

        mapOut.write("-----Tags Mapping-----\n")
        for tag, tag_num in tags_dict.items():
            mapOut.write(f"{tag} {tag_num}\n")



if __name__ == '__main__':

    # features_file, feature_vecs_file, feature_map_file = sys.argv[1:]
    features_file, feature_vecs_file, feature_map_file = "features_file", "feature_vecs_file", "feature_map_file"

    transform_features_2(features_file, feature_vecs_file, feature_map_file)


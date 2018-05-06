# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join
root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
     sys.path.insert(0, root_path)
# -------------------------

from collections import deque


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

            for feature in features:
                if features_dict.setdefault(feature, f) == f:
                    mapOut.write(f"{feature} {f}\n")
                    f += 1

            sorted_features = sorted((features_dict[feature] for feature in features))
            fOut.write("{tag} {features}\n".format(
                tag=tags_dict[tag],
                features=''.join((
                    f"{num}:1" for num in sorted_features
                ))
            ))

        mapOut.write("-----Tags Mapping-----\n")
        for tag, tag_num in tags_dict.items():
            mapOut.write(f"{tag} {tag_num}\n")



if __name__ == '__main__':

    # features_file, feature_vecs_file, feature_map_file = sys.argv[1:]
    features_file, feature_vecs_file, feature_map_file = "features_file", "feature_vecs_file", "feature_map_file"

    transform_features_2(features_file, feature_vecs_file, feature_map_file)


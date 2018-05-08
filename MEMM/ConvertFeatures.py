# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------

import sys
from MEMM.MEMMTaggers import MemmTagger
from utils.measuretime import measure


@measure
def main(featuresf, vectorf, mapfile):
    with open(vectorf, "w") as fOut:
        t = MemmTagger()
        for tagFeatures in t.fitFeaturesFromFile(featuresf):
            fOut.write(tagFeatures + '\n')
    t.saveParams(mapfile)


if __name__ == '__main__':
    features_file, feature_vecs_file, feature_map_file = sys.argv[1:4]
    main(features_file, feature_vecs_file, feature_map_file)

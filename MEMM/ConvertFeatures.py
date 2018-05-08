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

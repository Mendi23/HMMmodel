import sys
from MEMM.MEMMTaggers import MemmTagger
from utils.parsers import MappingParser
from time import time

if __name__ == '__main__':
    starttime = time()
    features_file, feature_vecs_file, feature_map_file = sys.argv[1:]

    with open(feature_vecs_file, "w") as fOut:
        t = MemmTagger()
        for tag, featVector in t.fitFeatures(features_file):
            fOut.write(MappingParser.TagVecToString(tag, featVector))
    MappingParser.saveDictsToFile([t.getFeaturesMapping(), t.getTagsMapping()], feature_map_file)
    print(time()-starttime)
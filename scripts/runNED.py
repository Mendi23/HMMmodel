import inspect

from sklearn.linear_model import LogisticRegression

from hmm2 import Taggers as hmm_tg
from MEMM import MEMMTaggers as hemm_tg, Features
from sys import argv

from utils.parsers import TagsParser

if __name__ == '__main__':
    train, dev  = argv[1:]

    x = hmm_tg.HmmModel(2)
    x.computeFromFile(train)
    hmm_greedy_tagger = hmm_tg.GreedyTagger(x)
    hmm_viterbi_tagger = hmm_tg.ViterbiTrigramTagger(x)

    t = hemm_tg.GreedyTagger(inspect.getmembers(Features, inspect.isfunction), LogisticRegression())
    x_train, y_train = [],[]
    featuresList = []
    for line in TagsParser().parseFile(train):
        for x in t.extractFeaturesFromTaggedLine(line):
            featuresList.append(x)

    t.fitFeaturesFromList(featuresList, transform=False)
    for x in featuresList:
        tag, features = t.transformTaggedToLiblinear(x)
        x_train.append(features)
        y_train.append(tag)
    t.fitModel(x_train, y_train)
    t.saveParams("dataFiles/NED_data", "dataFiles/NED_model")











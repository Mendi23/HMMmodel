from glob import glob
from sys import argv

import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from utils import MEMM_Taggers as hemm_tg
from hmm2 import hmm_Taggers as hmm_tg
from scripts_t.ner_eval import main_func
from utils.parsers import OutParser, TestParser, TagsParser


def trainModel():
    x_train, y_train = [], []
    featuresList = []
    for line in TagsParser().parseFile(train):
        for g in emm_greedy.extractTagFeatures(line):
            featuresList.append(g)

    emm_greedy.fitFeatures(featuresList, transform=False)
    for (tag, features) in featuresList:
        t, f = emm_greedy.transformTagged(tag, features)
        x_train.append(f)
        y_train.append(t)

    feat_count = emm_greedy.getNumOfFeatures()+1
    m = sp.lil_matrix((len(x_train),feat_count))

    for g,v in enumerate(x_train):
        m[g: v.size] = v

    emm_greedy.fitModel(m, y_train)
    emm_greedy.saveParams("dataFiles/NED_data", "dataFiles/NED_model")

def loadModels():
    emm_greedy.loadParams("dataFiles/NED_data", "dataFiles/NED_model")
    emm_viterbi.loadParams("dataFiles/NED_data", "dataFiles/NED_model")

if __name__ == '__main__':
    train, dev = argv[1:]
    x = hmm_tg.HmmModel(2)
    x.computeFromFile(train)
    hmm_greedy = hmm_tg.GreedyTagger(x)
    hmm_viterby = hmm_tg.ViterbiTrigramTagger(x)
    emm_greedy = hemm_tg.GreedyTagger(LogisticRegression())
    emm_viterbi = hemm_tg.ViterbiTrigramTagger(LogisticRegression())

    trainModel()
    loadModels()

    out = "testResult/NED/"
    outputFiles = ("hmm_greedy", "hmm_viterbi", "emm_greedy", "emm_viterbi")
    for i, tagger in enumerate((hmm_greedy, hmm_viterby, emm_greedy, emm_viterbi)):
        with OutParser(out+outputFiles[i]) as outF:
            for wordsLine in TestParser(splitWord = True).parseFile(dev):
                outF.printLine(tagger.tagLine(wordsLine))


    expected = list(TagsParser().parseTagsFromFile("DataSets/dev"))
    for file in glob("testResult/NED/*"):
        out = list(TagsParser().parseTagsFromFile(file))
        print(f"{file}:")
        main_func("DataSets/dev", file)
import inspect
from glob import glob

from sklearn.linear_model import LogisticRegression
from hmm2 import Taggers as hmm_tg
from MEMM import MEMMTaggers as hemm_tg, Features
from sys import argv

from scripts_t.ner_eval import main_func
from utils.parsers import OutParser, TestParser, TagsParser

if __name__ == '__main__':
    train, dev = argv[1:]

    # x = hmm_tg.HmmModel(2)
    # x.computeFromFile(train)
    # hmm_greedy = hmm_tg.GreedyTagger(x)
    # hmm_viterby = hmm_tg.ViterbiTrigramTagger(x)
    # emm_greedy = hemm_tg.GreedyTagger(inspect.getmembers(Features, inspect.isfunction),
    #     LogisticRegression())

    """
    x_train, y_train = [], []
    featuresList = []
    for line in TagsParser().parseFile(train):
        for x in emm_greedy.extractTagFeaturesFromTaggedLine(line):
            featuresList.append(x)

    emm_greedy.fitFeatures(featuresList, transform=False)
    for (tag, features) in featuresList:
        t, f = emm_greedy.transformTaggedToLiblinear(tag, features)
        x_train.append(f)
        y_train.append(t)

    feat_count = emm_greedy.getNumOfFeatures()+1
    m = sp.lil_matrix((len(x_train),feat_count))

    for i,v in enumerate(x_train):
        m[i: v.size] = v

    emm_greedy.fitModel(m, y_train)
    emm_greedy.saveParams("dataFiles/NED_data", "dataFiles/NED_model")
    """


    # emm_greedy.loadParams("dataFiles/NED_data", "dataFiles/NED_model")
    # out = "testResult/NED/"
    # outputFiles = ("hmm_greedy", "hmm_viterbi", "emm_greedy")
    # for i, tagger in enumerate((hmm_greedy, hmm_viterby, emm_greedy)):
    #     with OutParser(out+outputFiles[i]) as outF:
    #         for wordsLine in TestParser(splitWord = True).parseFile(dev):
    #             outF.printLine(tagger.tagLine(wordsLine))


    expected = list(TagsParser().parseTagsFromFile("DataSets/dev"))
    for file in glob("testResult/NED/*"):
        out = list(TagsParser().parseTagsFromFile(file))
        print(f"{file}:")
        main_func("DataSets/dev", file)
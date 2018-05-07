import inspect

from sklearn.linear_model import LogisticRegression

from hmm2 import Taggers as hmm_tg
from MEMM import MEMMTaggers as hemm_tg, Features
from sys import argv




if __name__ == '__main__':
    train, dev  = argv[1:]

    x = hmm_tg.HmmModel(2)
    x.computeFromFile(train)
    hmm_greedy_tagger = hmm_tg.GreedyTagger(x)
    hmm_viterbi_tagger = hmm_tg.ViterbiTrigramTagger(x)

    t = hemm_tg.MemmTagger(inspect.getmembers(Features, inspect.isfunction), LogisticRegression())
    t.fitFeatures(train, transform=False)
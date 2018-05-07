# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------


from hmm2.Taggers import ViterbiTrigramTagger
from utils.parsers import TestParser, OutParser
from utils.hmmModel import HmmModel
from sys import argv

if __name__ == '__main__':

    input_file, q_mle, e_mle, out_file = argv[1:5]

    x = HmmModel(2)
    x.loadTransitions(q_mle, e_mle)

    tagger = ViterbiTrigramTagger(x)
    with OutParser(out_file) as outF:
        for wordsLine in TestParser().parseFile(input_file):
            outF.printLine(tagger.tagLine(wordsLine))

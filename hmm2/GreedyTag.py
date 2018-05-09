# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join

from scripts_t.measuretime import measure

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------

from hmm2.hmm_Taggers import GreedyTagger
from utils.parsers import TestParser, OutParser
from utils.hmmModel import HmmModel

@measure
def main(inputf, q_file, e_file, outfile):
    x = HmmModel(2)
    x.loadTransitions(q_file, e_file)
    tagger = GreedyTagger(x)
    with OutParser(outfile) as outF:
        for wordsLine in TestParser().parseFile(inputf):
            outF.printLine(tagger.tagLine(wordsLine))

if __name__ == '__main__':

    input_file, q_mle, e_mle, out_file = sys.argv[1:5]
    main(input_file, q_mle, e_mle, out_file)

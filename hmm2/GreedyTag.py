# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join
root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
     sys.path.insert(0, root_path)
# -------------------------


from hmm2.hmmTagger import GreedyTagger
from parsers import TestParser, OutParser


if __name__ == '__main__':
    """ command line: 
        GreedyTag.py input_file_name q_mle_filename e_mle_filename output_file_name extra_file_name
    """
    from hmmModel import HmmModel
    from sys import argv

    input_file, q_mle, e_mle, out_file, extra = argv[1:]

    x = HmmModel(2)
    x.loadTransitions(q_mle, e_mle)

    tagger = GreedyTagger(x)
    with OutParser(out_file) as outF:
        for wordsLine in TestParser().parseFile(input_file):
            outF.printLine(tagger.tagLine(wordsLine))


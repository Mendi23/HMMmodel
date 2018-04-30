from time import time

from hmmTagger import GreedyTagger
from parsers import TestParser, OutputParser


if __name__ == '__main__':
    """ command line: 
        GreedyTag.py input_file_name q_mle_filename e_mle_filename output_file_name extra_file_name
    """
    from hmmModel import HmmModel
    from sys import argv

    input_file, q_mle, e_mle, out_file, extra = argv[1:]

    starttime = time()
    x = HmmModel(2)
    x.loadTransitions(q_mle, e_mle)

    tagger = GreedyTagger(x)
    with OutputParser(out_file) as outF:
        for wordsLine in TestParser().parseFile(input_file):
            out = tagger.tagLine(wordsLine)
            for word, tag in out:
                outF.append(word, tag)
            outF.breakLine()

    print("total {}s".format(time() - starttime))

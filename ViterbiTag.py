from time import time

from hmmTagger import ViterbiTagger
from parsers import TestParser, OutParser

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

    tagger = ViterbiTagger(x)
    with OutParser(out_file) as outF:
         # t = "`` Such agency ` self-help ' borrowing is unauthorized and expensive , far more expensive than direct Treasury borrowing , '' said Rep. Fortney Stark ( D. , Calif. ) , the bill 's chief sponsor ."
        # inp = t.split(" ")
        # out = tagger.tagLine(inp)
        # print (out)
        for wordsLine in TestParser().parseFile(input_file):
            outF.printLine(tagger.tagLine(wordsLine))
    print("total {}s".format(time() - starttime))

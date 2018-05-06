from collections import defaultdict, Counter, deque
from sys import argv
import matplotlib.pyplot as plt
import numpy as np

from hmmModel import HmmModel
from parsers import TagsParser, OutParser, TestParser
from testScore import main as testScore
from hmm2.Taggers import ViterbiTrigramTagger, GreedyTagger


def run_viterbi(infile, outfile, x, QH):
    tagger = ViterbiTrigramTagger(x, QHyperParam=QH)
    with OutParser(outfile) as outF:
        for wordsLine in TestParser().parseFile(infile):
            outF.printLine(tagger.tagLine(wordsLine))


if __name__ == '__main__':
    outfile = "viterbi_train_out.txt"
    x = HmmModel(2)
    x.loadTransitions("q.mle", "e.mle")
    #QH = x.deleted_interpolation()
    #print(str(QH))
    run_viterbi("DataSets/ass1-tagger-test-input", outfile, x, None)

    out = list(TagsParser().parseAllFromFile(outfile))
    expected = list(TagsParser().parseAllFromFile("DataSets/ass1-tagger-test"))

    diff = Counter()
    diffs = deque()
    diff_exists = Counter()
    diff_nonexists = Counter()
    n = 0
    for i, (a, b) in enumerate(zip(out, expected)):
        n += 1
        if a[1] != b[1]:
            diff[(b[1], a[1])] += 1
            diffs.append(i)
            if x.wordExists(a[0]):
                diff_exists[a[0]] += 1
            else:
                diff_nonexists[a[0]] += 1

    print(f"total precision: {1-sum(diff.values())/float(n)}")
    print(f"all diff: {sum(diff.values())}. exists: {sum(diff_exists.values())}. nonexists: {sum(diff_nonexists.values())}.")
    print(f"diff: {diff.most_common(10)}.\nexists: {diff_exists.most_common(10)}.\nnonexists: {diff_nonexists.most_common(10)}.")

    # ------------------------------------------
    gg = GreedyTagger(x)
    act_e, act_q, exp_e, exp_q = deque(), deque(), deque(), deque()
    def kukuriku ():
        for i in diffs:
            actualtag, expectedtag, word = out[i][1], expected[i][1], out[i][0]

            actual_q = gg._calcQ(tuple(t[1] for t in out[i - 2:i + 1]))
            actual_e = gg._calcE(word, actualtag)
            expected_q = gg._calcQ(tuple(t[1] for t in expected[i - 2:i + 1]))
            expected_e = gg._calcE(word, expectedtag)

            act_e.append(actual_e)
            act_q.append(actual_q)
            exp_e.append(expected_e)
            exp_q.append(expected_q)
    kukuriku()

    act_e = np.array(act_e)
    act_q = np.array(act_q)
    exp_e = np.array(exp_e)
    exp_q = np.array(exp_q)
    hefreshim = act_e * act_q - exp_e * exp_q

    print(np.average(act_e - exp_e))
    print(np.median(act_e - exp_e))
    exit()
    vec = np.vectorize(lambda x: 1 if x > 0 else -1)

    # plt.subplot(211)
    # plt.plot(sorted(hefreshim), label="hefreshim")
    # plt.legend()

    # plt.plot(np.full(hefreshim.shape, np.average(hefreshim)), label="hef_avg")
    # plt.subplot(212)
    # plt.plot(act_q - exp_q, label="Q")
    # plt.legend()
    # plt.subplot(122)
    # plt.plot(act_e - exp_e, label="E")
    plt.plot(sorted(vec(act_e - exp_e)), label="E")
    plt.plot(sorted(vec(act_q - exp_q)), label="Q")
    plt.legend()
    plt.show()




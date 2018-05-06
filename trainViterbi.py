from collections import defaultdict, Counter
from sys import argv

from hmmModel import HmmModel
from parsers import TagsParser, OutParser, TestParser
from testScore import main as testScore
from hmm2.Taggers import  ViterbiTrigramTagger

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
    diff_exists = Counter()
    diff_nonexists = Counter()
    n = 0
    for a, b in zip(out, expected):
        n += 1
        if a[1] != b[1]:
            diff[b[1]] += 1
            if x.wordExists(a[0]):
                diff_exists[a[0]] += 1
            else:
                diff_nonexists[a[0]] += 1

    print(f"total precision: {1-sum(diff.values())/float(n)}")
    print(f"all diff: {sum(diff.values())}. exists: {sum(diff_exists.values())}. nonexists: {sum(diff_nonexists.values())}.")
    print(f"diff: {diff.most_common(10)}.\nexists: {diff_exists.most_common(10)}.\nnonexists: {diff_nonexists.most_common(10)}.")


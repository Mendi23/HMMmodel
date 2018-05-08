from utils.parsers import TagsParser
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from glob import glob

def getScore (out, expected):
    binazier = MultiLabelBinarizer()
    binazier.fit(set(expected + out))
    return accuracy_score(binazier.transform(out), binazier.transform(expected))

if __name__ == '__main__':
    """ command line: 
        output_file, expected_result_file
    """
    expected = list(TagsParser().parseTagsFromFile("DataSets/ass1-tagger-test"))
    for file in glob("testResult/*.txt"):
        out = list(TagsParser().parseTagsFromFile(file))
        score = getScore(out, expected)
        print(f"{file}: {score}")

from utils.parsers import TagsParser
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer


def main (outputfile, expectedfile):
    out = list(TagsParser().parseTagsFromFile(outputfile))
    expected = list(TagsParser().parseTagsFromFile(expectedfile))
    # print(len(expected)-len(out))

    binazier = MultiLabelBinarizer()
    binazier.fit(set(expected + out))

    # s1 = classification_report(binazier.transform(out), binazier.transform(expected))

    score = accuracy_score(binazier.transform(out), binazier.transform(expected))
    return score

if __name__ == '__main__':
    """ command line: 
        output_file, expected_result_file
    """
    from sys import argv

    output_file, expected_result_file = argv[1:]

    score = main(output_file, expected_result_file)
    print(score)

from parsers import TagsParser
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
if __name__ == '__main__':
    """ command line: 
        output_file, expected_result_file
    """
    from sys import argv

    output_file, expected_result_file = argv[1:]


    out = list(TagsParser().parseTagsFromFile(output_file))
    expected = list(TagsParser().parseTagsFromFile(expected_result_file))
    print(len(expected)-len(out))


    binazier = MultiLabelBinarizer()
    binazier.fit(set(expected + out))

    #s1 = classification_report(binazier.transform(out), binazier.transform(expected))

    score = accuracy_score(binazier.transform(out), binazier.transform(expected))
    print(score)




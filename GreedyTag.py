


if __name__ == '__main__':
    from hmmModel import HmmModel
    from sys import argv

    x = HmmModel(2)
    x.loadTransitions(argv[1], argv[2])
    print(x.getE("an", "DT"))
    print(x.getQ(("NNP", "POS", "NNP"), (0.2, 0.2, 0.6)))
    m = 2
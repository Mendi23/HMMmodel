


if __name__ == '__main__':
    from hmmModel import HmmModel
    from sys import argv

    x = HmmModel(2)
    x.loadTransitions(argv[1], argv[2])
    m = 2
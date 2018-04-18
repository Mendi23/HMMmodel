
if __name__ == '__main__':
    from hmmModel import *
    from sys import argv

    x = HmmModel(2)
    x.computeFromFile(argv[1])
    x.writeQ(argv[2])
    x.writeE(argv[3])



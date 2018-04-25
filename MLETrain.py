from time import time

if __name__ == '__main__':
    from hmmModel import HmmModel
    from sys import argv

    starttime = time()

    x = HmmModel(2)
    x.computeFromFile(argv[1])
    x.writeQ(argv[2])
    x.writeE(argv[3])

    print("total {}s".format(time() - starttime))


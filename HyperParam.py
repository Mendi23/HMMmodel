"""
Hyper Parameters need Tuning:
 ,signatures
reComputeUnknown, newThreshold
getQ, hyperParam
"""

if __name__ == '__main__':
    from hmmModel import HmmModel
    from sys import argv

    x = HmmModel(2)
    x.computeFromFile(argv[1])


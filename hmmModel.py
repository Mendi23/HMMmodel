from collections import Counter
from itertools import product


class TransitionTable:

    def __init__(self, k = 3, items = None):
        self.k = k
        self.counter = Counter()
        if items:
            self.addFromList(items)

    def addFromList(self, items):
        itemsLen = len(items)
        x = product(range(itemsLen), range(1, self.k + 1))
        y = filter(lambda p: p[0]+p[1] <= itemsLen, x)
        self.counter += Counter([tuple(items[i:i + j]) for i, j in y])
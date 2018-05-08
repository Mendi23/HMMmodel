from collections import namedtuple, defaultdict
from itertools import product

import numpy as np


class ViterbiTrigramTaggerAbstract:
    TagVal = namedtuple("TagVal", "prev tag val")
    zeroTagVal = TagVal(None, "empty", -np.inf)

    @staticmethod
    def TagValVal(tagVal):
        return tagVal.val

    def __init__(self, startTag, _getPossibleTags, _getCellVal):
        """

        :type _getCellVal: function(line: [], i: int, tagsTriplet: tuple[3]) -> logaritmic probability
        :type _getPossibleTags: function(line: [], i: int) -> iter
        """
        self.startTag = startTag
        self._getPossibleTags = _getPossibleTags
        self._getCellVal = _getCellVal

    def tagLine(self, line):
        if not line:
            return None
        lineLength = len(line)
        vTable = [
            defaultdict(lambda: defaultdict(lambda: self.zeroTagVal))
            for _ in range(lineLength + 1)
        ]

        possibleTs = [self.startTag]
        possibleRs = [self.startTag]
        vTable[0][self.startTag][self.startTag] = self.TagVal(None, "start", np.log(1.0))

        maxTagVal = self.zeroTagVal
        for i in range(len(line)):
            table_i = i + 1
            possibleIts = possibleTs
            possibleTs = possibleRs
            possibleRs = self._getPossibleTags(line, i)

            for t, r in product(possibleTs, possibleRs):
                possibleValues = (self._calcVTableCell(vTable[table_i - 1][it][t], (it, t, r), line, i)
                                  for it in possibleIts)

                cell = max(possibleValues, key=self.TagValVal)
                if not np.isneginf(cell.val):
                    vTable[table_i][t][r] = cell

                if i == lineLength - 1:
                    maxTagVal = max(maxTagVal, cell, key=self.TagValVal)

        output = []
        self._appendSelectedTags(maxTagVal, line, len(line) - 1, output)
        return output

    def _calcVTableCell(self, VCell, tagsTriplet, line, i):
        val = self._getCellVal(line, i, tagsTriplet)
        if val == -np.inf:
            return self.zeroTagVal

        return self.TagVal(VCell, tagsTriplet[-1], val + VCell.val)

    def _appendSelectedTags(self, tagVal, line, i, output):
        if i > 0 and tagVal.prev:
            self._appendSelectedTags(tagVal.prev, line, i - 1, output)
        output.append((line[i], tagVal.tag))
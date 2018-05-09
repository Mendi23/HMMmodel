from collections import namedtuple, defaultdict, deque
from functools import total_ordering
from itertools import product

import numpy as np


@total_ordering
class TagVal:
    def __init__(self, prev, tag, val):
        self.prev = prev
        self.tag = tag
        self.val = val

    def __eq__(self, other):
        return (self.val, self.tag) == (other.val, other.tag)

    def __lt__(self, other):
        return self.val < other.val

    def __repr__(self):
        return f"TagVal({self.val}, {self.tag})"


class ViterbiTrigramTaggerAbstract:
    zeroTagVal = TagVal(None, "empty", -np.inf)

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
        vTable = [{} for _ in range(lineLength + 1)]

        possibleTs = [self.startTag]
        possibleRs = [self.startTag]
        vTable[0][self.startTag] = {self.startTag: TagVal(None, "start", np.log(1.0))}

        maxTagVal = self.zeroTagVal
        for i in range(len(line)):
            table_i = i + 1
            possibleIts = possibleTs
            possibleTs = possibleRs
            possibleRs = self._getPossibleTags(line, i)

            for t in possibleTs:
                result_dict = {}
                realPossibleRs = deque(possibleRs)
                for r in possibleRs:
                    possibleValues = (self._calcVTableCell(vTable[table_i - 1][it][t], (it, t, r), line, i)
                                      for it in possibleIts)

                    cell = max(possibleValues)
                    if cell.val != self.zeroTagVal.val:
                        result_dict[r] = cell
                    else:
                        realPossibleRs.remove(r)

                    if i == lineLength - 1:
                        maxTagVal = max(maxTagVal, cell)

                vTable[table_i][t] = result_dict
                possibleRs = realPossibleRs

        output = []
        self._appendSelectedTags(maxTagVal, line, lineLength - 1, output)
        return output

    def _calcVTableCell(self, VCell, tagsTriplet, line, i):
        val = self._getCellVal(line, i, tagsTriplet)

        return TagVal(VCell, tagsTriplet[-1], val + VCell.val)

    def _appendSelectedTags(self, tagVal, line, i, output):
        if i > 0 and tagVal.prev:
            self._appendSelectedTags(tagVal.prev, line, i - 1, output)
        output.append((line[i], tagVal.tag))

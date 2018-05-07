# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join
from MEMM.MEMMTaggers import MemmTagger

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------

import MEMM.Features as Features
import parsers
import inspect


if __name__ == '__main__':
    inputfilename, outputfilename = sys.argv[1:]
    t = MemmTagger(inspect.getmembers(Features, inspect.isfunction))

    with open(outputfilename, "w") as output:
        for line in parsers.TagsParser().parseFile(inputfilename):
            tag, features = t.extractFeaturesFromTaggedLine(line)
            output.write(f"{parsers.MappingParser.TagFeatToString(tag, features)}\n")

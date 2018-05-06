# for importing local modules
import sys
from inspect import currentframe, getfile as i_getfile
from os.path import realpath, abspath, split as p_split, join as p_join

root_path = realpath(abspath(p_join(p_split(i_getfile(currentframe()))[0], "..")))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
# -------------------------

if __name__ == '__main__':
    from hmmModel import HmmModel
    from sys import argv

    inputfilename, qfilename, efilename = argv[1:]
    x = HmmModel(2)
    x.computeFromFile(inputfilename)
    x.writeQ(qfilename)
    x.writeE(efilename)

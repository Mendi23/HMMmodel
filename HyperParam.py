
"""
Hyper Parameters need Tuning:
 ,signatures
reComputeUnknown, newThreshold
getQ, hyperParam
"""
from hmmModel import HmmModel

x = HmmModel(2)
x.loadTransitions("./q.mle", "./e.mle")
print(x.getE("an", "DT"))
print(x.getQ("NNP", "POS", "NNP"))

class ParametersLearning:
    pass
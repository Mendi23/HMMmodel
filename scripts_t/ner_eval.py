import codecs

def read_data(fname):
    for line in codecs.open(fname):
        line = line.strip().split()
        tagged = [x.rsplit("/",1) for x in line]
        yield tagged


def normalize_bio(tagged_sent):
    last_bio, last_type = "O","O"
    normalized = []
    for word, tag in tagged_sent:
        if tag == "O": tag = "O-O"
        bio,typ = tag.split("-",1)
        if bio=="I" and last_bio=="O": bio="B"
        if bio=="I" and last_type!=typ: bio="B"
        normalized.append((word,(bio,typ)))
        last_bio,last_type=bio,typ
    return normalized

def compare_accuracy(gold, pred):
    assert(len(gold)==len(pred))
    correct = 0.0
    total = 0.0
    for gold_sent, pred_sent in zip(gold, pred):
        assert(len(gold_sent)==len(pred_sent))
        gws = [w for w,t in gold_sent]
        pws = [w for w,t in pred_sent]
        assert(gws==pws)
        gtags = [t for w,t in gold_sent]
        ptags = [t for w,t in pred_sent]
        correct += sum([1 if g==p else 0 for g,p in zip(gold_sent, pred_sent)])
        total += len(gold_sent)
    return correct/total

def get_entities(sent):
    ent=[]
    for i,(word,tag) in enumerate(sent):
        bio,typ=tag
        if bio=="B":
            if ent: yield tuple(ent)
            ent=[]
            ent.append(i)
            ent.append(typ)
            ent.append(word)
        if bio=="I":
            ent.append(word)
        if bio=="O":
            if ent: yield tuple(ent)
            ent=[]
    if ent: yield tuple(ent)

def main_func(gold_file, pred_file):
    gold_data = [normalize_bio(tagged) for tagged in read_data(gold_file)]
    pred_data = [normalize_bio(tagged) for tagged in read_data(pred_file)]

    assert(len(gold_data)==len(pred_data))

    gold_entities = set()
    for i,sent in enumerate(gold_data):
        for entity in get_entities(sent):
            gold_entities.add((i,entity))

    pred_entities = set()
    for i,sent in enumerate(pred_data):
        for entity in get_entities(sent):
            pred_entities.add((i,entity))

    acc = compare_accuracy(gold_data, pred_data)
    print ("Accuracy: %f" % acc)

    prec = len(gold_entities.intersection(pred_entities)) / (float(len(pred_entities)) or 1)
    rec  = len(gold_entities.intersection(pred_entities)) / (float(len(gold_entities)) or 1)
    print ("All-types \tPrec:%s Rec:%s" % (prec, rec))

    types = set([e[1][1] for e in gold_entities]) - set(["O"])
    for t in types:
        gents = set([e for e in gold_entities if e[1][1]==t])
        pents = set([e for e in pred_entities if e[1][1]==t])
        prec = len(gents.intersection(pents)) / (float(len(pents)) or 1)
        rec  = len(gents.intersection(pents)) / (float(len(gents)) or 1)
        print ("%10s \tPrec:%s Rec:%s" % (t, prec, rec))






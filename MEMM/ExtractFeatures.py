import MEMM.Features as Features
import parsers
import inspect

if __name__ == '__main__':
    allFeatures = inspect.getmembers(Features, inspect.isfunction)

    print("Feature names:", end="\n  ")
    print(', '.join((feature[0] for feature in allFeatures)))

    input("press enter to print all lines...")

    for line in parsers.TagsParser().parseFile("../DataSets/ass1-tagger-train"):
        words, tags = tuple(zip(*line))

        for i in range(len(line)):
            featuresvals = filter(lambda v: v["val"],
                                  ({"name": feature[0], "val": feature[1](words, tags, i)}
                                   for feature in allFeatures))

            print("{tag} {features}".format(
                tag=tags[i],
                features=' '.join(v["name"]+"="+v["val"] for v in featuresvals)))

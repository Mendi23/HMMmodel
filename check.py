

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



cv = TfidfVectorizer("file", analyzer = "word", token_pattern = "/[^\s]+", ngram_range = (1,3))
a = cv.build_analyzer()
with open("./DataSets/ass1-tagger-train") as f:
    x = cv.fit_transform((f,))
    y = cv.get_feature_names()
    print(y)
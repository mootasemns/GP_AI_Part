from sense2vec import Sense2Vec
from utils.utils import *
class Sense2VecModel:
    def __init__(self, model_path):
        self.model = Sense2Vec().from_disk(model_path)

    def get_words(self, word, topn, question):
        output = []
        try:
            sense = self.model.get_best_sense(word, senses=["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WORK OF ART", "FAC", "GPE", "NUM", "FACILITY"])
            most_similar = self.model.most_similar(sense, n=topn)
            output = filter_same_sense_words(sense, most_similar)
        except:
            output = []

        threshold = 0.6
        final = [word]
        checklist = question.split()
        for x in output:
            if get_highest_similarity_score(final, x) < threshold and x not in final and x not in checklist:
                final.append(x)

        return final[1:]

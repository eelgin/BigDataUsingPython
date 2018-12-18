#%matplotlib inline
#import pandas as pd
#import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from mrjob.job import MRJob
from mrjob.step import MRStep
import re

WORD_RE = re.compile(r"[\w']+")

class BigramFrequencyCount(MRJob):

    MRJob.SORT_VALUES = True

    def steps(self):
        return [
            MRStep(mapper = self.mapper_1,
                   reducer = self.reducer_1),
            MRStep(mapper = self.mapper_2,
                   reducer = self.reducer_2)
        ]

    def mapper_1(self, _, line):
        wprev = None;
        for word in WORD_RE.findall(line):
            word = word.lower()
            if (wprev is None):
                wprev = word
                continue
            yield (wprev, word), 1
            wprev = word

    def reducer_1(self, word, counts):
        yield word, sum(counts)

    def mapper_2(self, word, countTotal):
        yield None, ("%07.0f"%float(countTotal), word)

    def reducer_2(self, _, word_count_pairs):
        for c in word_count_pairs:
            yield c[1], c[0]


if __name__ == '__main__':
    BigramFrequencyCount.run()

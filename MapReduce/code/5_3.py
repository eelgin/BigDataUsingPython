# 5_3 code

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
        wprev1 = None;
        wprev2 = None;
        for word in WORD_RE.findall(line):
            word = word.lower()
            if (wprev1 is None or
                wprev2 is None):
                wprev2 = wprev1
                wprev1 = word
                continue

            order = [word,wprev1,wprev2]
            wprev2 = wprev1
            wprev1 = word
            order.sort()

            yield (order[0], order[1], order[2]), 1


    def reducer_1(self, words, counts):
        yield words, sum(counts)

    def mapper_2(self, words, countTotal):
        yield None, ("%07.0f"%float(countTotal), words)

    def reducer_2(self, _, word_count_pairs):
        for c in word_count_pairs:
            yield c[1], c[0]


if __name__ == '__main__':
    BigramFrequencyCount.run()

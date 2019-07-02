
import numpy
import gzip
import os


PAD = 'PADDING_TOKEN'
UNK = 'UNKNOWN_TOKEN'
UNK_IX = 1

def invert_index(xs):
    return {x: i for i, x in enumerate(xs)}

class DSM(object):
    def __init__(self, words, vecs):
        self.w2i = invert_index(words)
        self.i2w = words
        self.m = vecs

    def __getitem__(self, item):
        if type(item) is list:
            return self.m[[self.w2i[i] for i in item]]
        else:
            return self.m[self.w2i[item]]

    def get(self, item):
        if type(item) is list:
            return self.m[[self.w2i.get(i, UNK_IX) for i in item]]
        else:
            return self.m[self.w2i.get(item, UNK_IX)]

    def __contains__(self, item):
        return item in self.w2i


    @property
    def shape(self):
        return self.m.shape

    @staticmethod
    def read(path, restrict=None, dtype=numpy.float32):
        if restrict is not None:
            def check(word):
                return word in restrict
        else:
            def check(word):
                return True

        words = [PAD, UNK]
        vecs = [None, None]

        def addpair(wv):
            word = wv[0]
            if check(word):
                vecs.append(
                    numpy.array([float(num) for num in wv[1:]], dtype=dtype)
                )
                words.append(word)

        with open(path, 'rt') as handle:
            wv = next(handle).strip().split(" ")
            if len(wv) > 2:
                addpair(wv)
            for line in handle:
                wv = line.strip().split(" ")
                addpair(wv)

        dim = len(wv) - 1
        vecs[0] = numpy.zeros(dim).astype(dtype) #Zero vector for 'PAD' word (???)
        vecs[1] = numpy.random.uniform(-0.25, 0.25, dim).astype(dtype) #Random vector for 'UNK' word (???)

        return DSM(words, numpy.array(vecs))


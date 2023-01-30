import numpy as np


class datagen:
    def init(self):
        return

    def nor_gen(self, mean, sigma, size):
        ranint = np.random.normal(mean, sigma, size)
        qdata = ranint
        ranint.sort()
        return ranint, qdata

    def uni_gen(self, low, high, size):
        rannum = np.random.randint(low, high, size)
        qdata = rannum
        rannum.sort()
        return rannum, qdata

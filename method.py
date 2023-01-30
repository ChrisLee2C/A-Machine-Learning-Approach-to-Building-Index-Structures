import numpy as np
from scipy.stats import norm
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
import pwlf


class linearsearch:
    def __init__(self):
        return

    def search(self, arr, x):
        for i in range(len(arr)):
            if arr[i] == x:
                return i
        return -1


class binarysearch:
    def __init__(self):
        return

    def search(self, arr, l, r, x):
        countb = 1
        while l <= r:
            mid = l + (r - l) // 2
            if arr[mid] == x:
                return mid, countb
            elif arr[mid] < x:
                l = mid + 1
                countb += 1
            else:
                r = mid - 1
                countb += 1
        return -1, countb


class hashtable:
    def __init__(self):
        return

    def gen_table(self, arr):
        hashtable = {}
        for i in range(len(arr)):
            key = hash(arr[i])
            val = arr[i]
            hashtable.update({key: val})
        return hashtable

    def search(self, arr, x):
        hashtable = self.gen_table(arr)
        if hash(x) in hashtable:
            return 1
        else:
            return -1


class trickmethod:
    def __init__(self):
        return

    def predict(self, size, x):
        position = size * x / 1000000000
        return position

    def search(self, size, arr, x):
        starting_position = self.predict(size, x)
        for i in range(int(starting_position), len(arr)):
            if arr[i] == x:
                return i
        return -1


class linearregression:
    def __init__(self):
        return

    def crtCDF(self, x):
        if type(x) == np.ndarray:
            loc = x.mean()
            scale = x.std()
            N = x.size
            pos = norm.cdf(x, loc, scale) * N
            return pos
        else:
            print("Wrong Type! x must be np.ndarray ~")
            return

    def build(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[1]),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model

    def train(self, model, arr):
        x = arr
        y = self.crtCDF(x)
        model.fit(x, y, epochs=100, batch_size=32, verbose=0)
        x = np.reshape(x, (-1, 1))
        model = LinearRegression()
        model.fit(x, y)

    def search(self, model, querydata, arr, size):
        target = model.call(tf.convert_to_tensor([querydata]))
        counts = 0
        if target >= size:
            target = size-1
        elif target < 0:
            target = 0
        if arr[int(target)] != querydata:
            if arr[int(target)] < querydata:
                for i in range(int(target)+1, len(arr)):
                    counts += 1
                    if arr[i] == querydata:
                        return i, counts
                return -1, counts
            if arr[int(target)] > querydata:
                for i in range(int(target)-1, 0, -1):
                    counts += 1
                    if arr[i] == querydata:
                        return i, counts
                return -1, counts
        else:
            return target, counts + 1


class piecewiselinearregression:
    def __init__(self):
        return

    def crtCDF(self, x):
        if type(x) == np.ndarray:
            loc = x.mean()
            scale = x.std()
            N = x.size
            pos = norm.cdf(x, loc, scale) * N
            return pos
        else:
            print("Wrong Type! x must be np.ndarray ~")
            return

    def train(self, arr):
        x = arr
        y = self.crtCDF(x)
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        my_pwlf.fit(3)
        return my_pwlf

    def search(self, model, querydata, arr, size):
        target = model.predict([querydata])
        counts = 0
        if target >= size:
            target = size - 1
        elif target < 0:
            target = 0
        if arr[int(target)] != querydata:
            if arr[int(target)] < querydata:
                for i in range(int(target) + 1, len(arr)):
                    counts += 1
                    if arr[i] == querydata:
                        return i, counts
                return -1, counts
            if arr[int(target)] > querydata:
                for i in range(int(target) - 1, 0, -1):
                    counts += 1
                    if arr[i] == querydata:
                        return i, counts
                return -1, counts
        else:
            return target, counts + 1
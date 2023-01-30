import timeit
from method import *
from datagen import *

arr = []
querydata = None
countb = counts = 0

low = 1
high = 10000000
size = 10000
datatype = datagen()
start = timeit.default_timer()
arr,querydata = datatype.nor_gen(low, high, size)
stop = timeit.default_timer()
file = open("Dataset.txt", "a")
file.write(str(arr))
file.write("\n")
file.close()
model = linearregression()
bmodel = model.build()
model.train(bmodel, arr)

model2 = piecewiselinearregression()
pmodel = model2.train(arr)
for iter in range(1000):
    methodl = linearsearch()
    start = timeit.default_timer()
    lposition = methodl.search(arr, querydata[iter])
    stop = timeit.default_timer()
    ltime = stop - start
    lfile = open("LinearSearchTime.txt", "a")
    lfile.write(str(ltime))
    lfile.write("\n")
    lfile.close()
    l2file = open("LinearSearchIterations.txt", "a")
    if lposition == -1:
        l2file.write(str(len(arr)))
    else:
        l2file.write(str(lposition+1))
    l2file.write("\n")
    l2file.close()

    methodb = binarysearch()
    start = timeit.default_timer()
    bposition, countb = methodb.search(arr, 0, len(arr) - 1, querydata[iter])
    stop = timeit.default_timer()
    btime = stop - start
    bfile = open("BinarySearchTime.txt", "a")
    bfile.write(str(btime))
    bfile.write("\n")
    bfile.close()
    b2file = open("BinarySearchIterations.txt", "a")
    b2file.write(str(countb))
    b2file.write("\n")
    b2file.close()

    methodh = hashtable()
    start = timeit.default_timer()
    hposition = methodh.search(arr, querydata[iter])
    stop = timeit.default_timer()
    htime = stop - start
    hfile = open("HashtableSearchTime.txt", "a")
    hfile.write(str(htime))
    hfile.write("\n")
    hfile.close()

    methodt = trickmethod()
    start = timeit.default_timer()
    tposition = methodt.search(len(arr), arr, querydata[iter])
    stop = timeit.default_timer()
    ttime = stop - start
    tfile = open("TrickMethodSearchTime.txt", "a")
    tfile.write(str(ttime))
    tfile.write("\n")
    tfile.close()

    start = timeit.default_timer()
    sposition, counts = model.search(bmodel, querydata[iter], arr, size)
    stop = timeit.default_timer()
    stime = stop - start
    sfile = open("LinearRegressionSearchTime.txt", "a")
    sfile.write(str(stime))
    sfile.write("\n")
    sfile.close()
    s2file = open("LinearRegressionSearchIterations.txt", "a")
    s2file.write(str(counts))
    s2file.write("\n")
    s2file.close()

    start = timeit.default_timer()
    pposition, countp = model2.search(pmodel, querydata[iter], arr,size)
    stop = timeit.default_timer()
    ptime = stop - start
    pfile = open("PiecewiseLinearRegressionSearchTime.txt", "a")
    pfile.write(str(ptime))
    pfile.write("\n")
    pfile.close()
    p2file = open("PiecewiseLinearRegressionSearchIterations.txt", "a")
    p2file.write(str(countp))
    p2file.write("\n")
    p2file.close()
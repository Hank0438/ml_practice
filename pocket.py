import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random

#網路上找的dataset 可以線性分割

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
data = pd.read_csv(SCRIPT_PATH + "/data/pla_data.txt", header=None, encoding='utf-8', delim_whitespace=True)
dataset = np.array(data)
dataset = np.insert(dataset, 0, 1, axis=1)
print(dataset.shape)
print(dataset)

#產生亂數序列
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

#判斷有沒有分類錯誤，並列印錯誤率

def check_error(w, dataset):
    result = None
    error = 0
    for i in randomly(dataset):
        x = i[:len(i)-1]
        s = i[len(i)-1]
        #將w轉置且和x取內積後判斷是正or負or零
        if int(np.sign(w.T.dot(x))) != s: 
            result =  x, s
            error += 1
    #print("error=%s/%s" % (error, len(dataset)) )
    return result

#PLA演算法實作

def pla(dataset,):
    #w = [0, 0, 0]
    w = np.zeros(len(dataset[0])-1)
    count = 0
    while check_error(w, dataset) is not None:
        x, s = check_error(w, dataset)
        w += 0.5 * s * x
        count += 1
    print("count:",count)
    return count

#執行

c = 0
for j in range(2000):
    random.seed(j)
    c += pla(dataset)
c /= 2000
print("avg count:",c)

#結果：40.xxxx
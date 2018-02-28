import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import math

#dataset 
SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
train_data = pd.read_csv(SCRIPT_PATH + "/data/pocket_train_data.txt", header=None, encoding='utf-8', delim_whitespace=True)
train_data = np.array(train_data)
train_data = np.insert(train_data, 0, 1, axis=1)
test_data = pd.read_csv(SCRIPT_PATH + "/data/pocket_test_data.txt", header=None, encoding='utf-8', delim_whitespace=True)
test_data = np.array(test_data)
test_data = np.insert(test_data, 0, 1, axis=1)
#print(train_data.shape)

#產生亂數序列
def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)

#判斷有沒有分類錯誤
def check_error(w, dataset):
    result = None
    error = 0
    for j in randomly(range(len(dataset))):
        i = dataset[j]
        x = i[:len(i)-1]
        s = i[len(i)-1]
        #將w轉置且和x取內積後判斷是正or負or零
        if int(np.sign(w.T.dot(x))) != s: 
            error += 1
            result =  x, s, error
    return result

#Pocket演算法實作
def pocket(dataset):
    #w = [0, 0, 0]
    w = np.zeros(len(dataset[0])-1)
    iter_count = 50
    opt_w = np.array(w)
    opt_error = math.inf
    while check_error(w, dataset) is not None:
        x, s, error = check_error(w, dataset)
        w += 0.5 * s * x
        if error < opt_error:
            opt_error = error
            opt_w = np.array(w)
        #print(opt_error)
        #print(opt_w)
        #限定update次數
        iter_count -= 1
        if iter_count <= 0: break
    return np.array(w)

#執行
c = 0
time = 2000
for t in range(time):
    #讓隨機的分布不同
    random.seed(t)
    opt_w = pocket(train_data)
    x, s, error = check_error(opt_w, test_data)
    error_rate = error/len(test_data)
    print(t)
    print("error_rate:", error_rate)
    c += error_rate
c /= time
print("avg error rate:", c)
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
data = pd.read_csv(SCRIPT_PATH + "/data/pla_data.txt", header=None, encoding='utf-8', delim_whitespace=True)
dataset = np.array(data)
#print(dataset.shape)
dataset = dataset.T
#print(dataset.shape)


def learn_perceptron(times=1000):
    w1,w2,w3,w4 = 1,1,1,1
    count = 0
    X1 = dataset[0]
    X2 = dataset[1]
    X3 = dataset[2]
    X4 = dataset[3]
    Y = dataset[4]
    for i in range(times):
        ERR = (w1*X1+w2*X2+w3*X3+w4*X4) * Y < 0
        if len([e for e in ERR if e != True]) > 0:
           err_x1,err_x2,err_x3,err_x4,err_y = X1[ERR][0],X2[ERR][0],X3[ERR][0],X4[ERR][0],Y[ERR][0]
           #更新w
           w1,w2,w3,w4 = (w1+err_y*err_x1),(w2+err_y*err_x2),(w3+err_y*err_x3),(w4+err_y*err_x4)
           count +=1
        else: 
           print("Complete!")
           break;
    print("count:",count)

learn_perceptron()
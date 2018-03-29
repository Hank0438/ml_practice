import numpy as np

def linearReg(total_E_in, total_E_out, datasize):
    #隨機生成訓練樣本，且排序
    x0 = np.ones(datasize)
    x1 = np.random.uniform(-1,1,datasize)
    x2 = np.random.uniform(-1,1,datasize)
    x = np.array([x0,x1,x2]).T
    #判斷是否在h=圓的範圍內，且隨機增加noise
    y = []
    for i in range(datasize):
        yi = np.sign(np.power(x1[i],2) + np.power(x2[i],2) - 0.6)
        if(np.random.random_sample() < 0.1):
            yi *= -1
        y.append(yi)
    y = np.array(y)
    #計算[1,x1,x2]的虛反
    x_pinv = np.linalg.pinv(x)
    w_lin = np.dot(x_pinv,y)
    #計算預測的y，且計算E_in
    y_hat = np.sign(np.dot(x, w_lin))
    err = 0
    for i in range(datasize):
        if(y[i] != y_hat[i]): err +=1
    #print("error rate:",err/datasize)
    
    return err/datasize

def main():
    total_E_in = 0.0
    total_E_out = 0.0
    datasize = 1000
    times = 1000
    for i in range(times):
        print(i)
        Ein = linearReg(total_E_in, total_E_out, datasize)
        total_E_in += Ein
        #total_E_out += Eout
    print("avg_E_in:",total_E_in/times)
    #print("avg_E_out:",total_E_out/times)

main()
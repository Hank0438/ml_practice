import numpy as np

def generateData(datasize):
    #隨機生成訓練樣本
    x0 = np.ones(datasize)
    x1 = np.random.uniform(-1,1,datasize)
    x2 = np.random.uniform(-1,1,datasize)
    x1_sqr = np.square(x1)
    x2_sqr = np.square(x2)
    x1x2 = []
    for i in range(datasize):
        x1x2.append(x1[i]*x2[i])
    x1x2 = np.array(x1x2)
    x = np.array([x0,x1,x2,x1x2,x1_sqr,x2_sqr]).T
    #判斷是否在h=圓的範圍內，且隨機增加noise
    y = []
    for i in range(datasize):
        yi = np.sign(np.power(x1[i],2) + np.power(x2[i],2) - 0.6)
        if(np.random.random_sample() < 0.1):
            yi *= -1
        y.append(yi)
    y = np.array(y)
    return x,y

def linearReg(datasize):
    x,y = generateData(datasize)
    #計算[1,x1,x2,...]的虛反
    x_pinv = np.linalg.pinv(x)
    w_lin = np.dot(x_pinv,y)
    #print(w_lin)
    #計算預測的y，且計算E_in
    y_hat = np.sign(np.dot(x, w_lin))
    err = 0
    for i in range(datasize):
        if(y[i] != y_hat[i]): err +=1
    errorRate = err/datasize
    #print("error rate:",errorRate)
    
    return w_lin,errorRate

def Eout(datasize, w_lin):
    x,y = generateData(datasize)
    #計算預測的y，且計算E_out
    y_hat = np.sign(np.dot(x, w_lin))
    err = 0
    for i in range(datasize):
        if(y[i] != y_hat[i]): err +=1
    errorRate = err/datasize
    #print("error rate:",errorRate)
    
    return errorRate

def main():
    total_E_out = 0
    datasize = 1000
    times = 1000
    for i in range(times):
        print(i)
        w,E_in = linearReg(datasize)
        E_out = Eout(datasize, w)
        total_E_out += E_out
    print(total_E_out/1000)

#linearReg(0,0,1000)
main()
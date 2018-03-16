#a) for each dimension i=1,2,⋯,d, find the best decision stump h(s,i,θ)
# using the one-dimensional decision stump algorithm that you have just implemented.
#b) return the "best of best"' decision stump in terms of Ein. 
# If there is a tie , please randomly choose among the lowest-Ein ones
#Run the algorithm on the Dtrain. Report the Ein of the optimal decision stump returned by your program. 
import numpy as np

#dataset 
#train有100筆，test有1000筆
train_data = np.loadtxt("C:/Users/USER/Desktop/ml_practice/data/stump_train_data.txt")
test_data = np.loadtxt("C:/Users/USER/Desktop/ml_practice/data/stump_test_data.txt")

def calculateError(x, y, s, theta, datasize):
    #给定输入集合和指定的hyphothesis计算对应的错误率  
    error = 0;  
    for i in range(datasize): 
        if(s * np.sign(x[i] - theta) != y[i]): error+=1  
    errorRate = error/datasize   
    return errorRate

def E_in(x, y, opt_s, opt_theta, datasize):
    min_errorRate = 1.0
    #s表示1為positive ray，-1為negative ray
    for s in [1,-1]:
        #[-1,1]中隨機(uniform)取20個點分隔為21個區間作為theta的取值區間
        #theta就是dichotomy分類的位置
        for i in range(datasize + 1):
            #theta小於最小
            if(i == 0): theta = x[i] - 1.0
            #theta大於最大  
            elif(i == datasize): theta = x[datasize-1] + 1.0
            #theta在兩者之間
            else: theta = (x[i-1] + x[i]) / 2.0  
            
            errorRate = calculateError(x, y, s, theta, datasize)
            #如果此hyphothesis的錯誤更小則替代
            if(errorRate < min_errorRate):  
                opt_s = s
                opt_theta = theta
                min_errorRate = errorRate  
    return min_errorRate, opt_s, opt_theta

def E_out(s, theta):
    opt_Eout = 1.0
    #對每一維度(xi)
    for i in range(len(test_data[0])-1):
        idx = np.argsort(test_data.T[i])
        x = test_data.T[i][idx]
        y = test_data.T[len(test_data[0])-1][idx]
        #s,theta是由training data產生，不必再計算s,theta
        Eout = calculateError(x, y, s, theta, len(x))
        if(Eout < opt_Eout):
            opt_Eout = Eout
    return opt_Eout

def decision_stump():
    opt_Ein = 1.0
    opt_s = 0
    opt_theta = 0
    #對每一維度(xi)
    for i in range(len(train_data[0])-1):
        #將每一維度sorting後的index序列保存，並用這序列排序x,y
        idx = np.argsort(train_data.T[i])
        x = train_data.T[i][idx]
        y = train_data.T[len(train_data[0])-1][idx]
        #計算Ein, s(正或負), theta(切在數線上的哪點)
        Ein, s, theta = E_in(x, y, 0, 0, len(x))
        #比較E_in挑選最佳的s,theta
        if(Ein < opt_Ein):
            opt_Ein = Ein
            opt_s = s
            opt_theta = theta
    return opt_Ein, opt_s, opt_theta

opt_Ein, opt_s, opt_theta = decision_stump()
Eout = E_out(opt_s, opt_theta)
print(Eout)

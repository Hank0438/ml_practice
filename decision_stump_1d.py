#(a) Generate x by a uniform distribution in [−1,1].
#(b) Generate y by f(x)=s(x) + noise 
#where s(x)=sign(x) and the noise flips the result with 20% probability.

#在[-1,1]种取20个点
#分隔为21个区间作为theta的取值区间，每种分类有42个hyphothesis
#枚举所有可能情况找到使E_in最小的hyphothesis，记录最小E_in
import numpy as np

def E_in(x, y, opt, datasize=20):
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
            
            #给定输入集合和指定的hyphothesis计算对应的错误率  
            error = 0;  
            for i in range(datasize): 
                if(s * np.sign(x[i] - theta) != y[i]): error+=1  
            errorRate = error/datasize 

            #如果此hyphothesis的錯誤更小則替代
            if(errorRate < min_errorRate):  
                opt = s, theta
                min_errorRate = errorRate  

    return min_errorRate, opt

def E_out(opt):
    s, theta = opt
    return 0.5 + 0.3*s*(np.absolute(theta)-1.0)

def decision_stump(total_E_in, total_E_out, datasize=20):
    #隨機生成訓練樣本，且排序
    x = np.random.uniform(-1,1,datasize)
    np.sort(x)
    y = np.sign(x)
    #隨機增加noise
    for i in range(datasize):
        if(np.random.random_sample() < 0.2):
            y[i] = y[i]*-1
    opt = 0,0
    Ein, opt = E_in(x,y,opt)
    Eout = E_out(opt)
    #print("E_in:",Ein)
    #print("E_out:",Eout)
    return Ein, Eout


total_E_in = 0.0
total_E_out = 0.0
for i in range(5000):
    print(i)
    Ein, Eout = decision_stump(total_E_in, total_E_out)
    total_E_in += Ein
    total_E_out += Eout
print("avg_E_in:",total_E_in/5000)
print("avg_E_out:",total_E_out/5000)
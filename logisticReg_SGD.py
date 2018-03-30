import numpy as np

#Implement the fixed learning rate stochastic gradient descent algorithm for logistic regression. 
#Run the algorithm with η=0.001 and T=2000.
#What is Eout(g) from your algorithm, evaluated using the 0/1 error on the test set?

#dataset 
#train有1000筆，test有3000筆
train_data = np.loadtxt("C:/Users/USER/Desktop/ml_practice/data/logistic_train_data.txt")
test_data = np.loadtxt("C:/Users/USER/Desktop/ml_practice/data/logistic_test_data.txt")

def SGD(n, w, dataset):
    #Instead of randomly choosing n in each iteration, 
    #please simply pick the example with the cyclic order n=1,2,…,N,1,2,….
    x,y = dataset[n][:-1], dataset[n][-1]
    s = -y*np.dot(w,x)
    theta = 1/(1 + np.exp(-s))
    sgd = theta*-y*x
    return sgd

def Eout(dataset, g):
    #計算預測的y，且計算E_out
    err = 0
    for i in dataset:
        x,y = i[:-1],i[-1]
        y_hat = np.sign(np.dot(x, g))
        if(y != y_hat): err +=1
    errorRate = err/len(dataset)
    #print("error rate:",errorRate)    
    return errorRate

def logisticReg(dataset, eta, times):
    #預設w_0為[0,...,0]
    w = np.zeros(len(train_data[0])-1)
    #更新w_t+1 = w_t + eta * gradient_descent
    for i in range(times):
        for j in range(len(dataset)):
            w = w - (eta * SGD(j, w, dataset))
    #print(w)
    E_out = Eout(test_data, w)
    print(E_out)
    return E_out

logisticReg(train_data, 0.001, 2000)
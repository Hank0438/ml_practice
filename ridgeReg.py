import numpy as np

def errorRate(w_reg, dataset):
    datasize = len(dataset)
    x0 = np.ones(datasize)
    x1 = dataset.T[0]
    x2 = dataset.T[1]
    y = dataset.T[2]
    x = np.array([x0,x1,x2])
    
    #compute predicted y_hat and use y_hat to calculate error_rate
    y_hat = np.sign(np.dot(w_reg.T, x))
    err = 0
    for i in range(datasize):
        if(y[i] != y_hat[i]): err +=1
    error_rate = err/(datasize/1.0)
    print("error_rate:", error_rate)
    return error_rate

def ridgeReg(lamda):
    #dataset
    #train=200*3,test=1000*3
    train_data = np.loadtxt("C:/Users/USER/Desktop/ml_practice/data/ridge_train_data.txt")
    test_data = np.loadtxt("C:/Users/USER/Desktop/ml_practice/data/ridge_test_data.txt")
    datasize = len(train_data)
    x0 = np.ones(datasize)
    x1 = train_data.T[0]
    x2 = train_data.T[1]
    y = train_data.T[2]
    x = np.array([x0,x1,x2])
    
    #compute w_reg  
    inv = np.linalg.inv( np.dot(x,x.T) + lamda*np.eye(3))
    w_reg = np.dot(inv,np.dot(x,y.T))
    
    #compute E_in and E_out
    E_in, E_out = 0, 0
    E_in = errorRate(w_reg, train_data)
    E_out = errorRate(w_reg, test_data)
    return E_in, E_out

ridgeReg(10)
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
    return error_rate

def ridgeReg(lamda):
    #dataset
    #train=200*3,test=1000*3
    train_data = np.loadtxt("C:/Users/USER/Desktop/ml_practice/data/ridge_train_data.txt")
    test_data = np.loadtxt("C:/Users/USER/Desktop/ml_practice/data/ridge_test_data.txt")
    split_train = train_data[:120]
    split_validate = train_data[120:]
    #print(split_train.shape, split_validate.shape)
    datasize = len(split_train)
    x0 = np.ones(datasize)
    x1 = split_train.T[0]
    x2 = split_train.T[1]
    y = split_train.T[2]
    x = np.array([x0,x1,x2])
    
    #compute w_reg  
    inv = np.linalg.inv( np.dot(x,x.T) + lamda*np.eye(3))
    w_reg = np.dot(inv,np.dot(x,y.T))
    
    #compute E_in and E_out
    E_train, E_val, E_out = 0, 0, 0
    E_train = errorRate(w_reg, split_train)
    E_val = errorRate(w_reg, split_validate)
    E_out = errorRate(w_reg, test_data)
    print("E_train, E_val, E_out:", E_train, E_val, E_out)
    return E_train, E_val, E_out

arr = np.arange(13)-10
print(arr)
for i in arr:
    if(i<0): test = 1.0/np.power(10,-1*i) 
    else: test = np.power(10,i)
    print(test)
    ridgeReg(test)

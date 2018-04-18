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
    train_data = np.loadtxt("C:/Users/USER/Desktop/ml_practice/data/ridge_train_data.txt")
    accum_E_val = 0.0
    v_fold = 5
    for i in range(v_fold):
        idx = i*40
        split_train = np.concatenate((train_data[0:idx], train_data[idx+40:200]),axis=0)
        split_validate = train_data[idx:idx+40]
        
        #seperate x and y
        datasize = len(split_train)
        x0 = np.ones(datasize)
        x1 = split_train.T[0]
        x2 = split_train.T[1]
        y = split_train.T[2]
        x = np.array([x0,x1,x2])
    
        #compute w_reg  
        inv = np.linalg.inv( np.dot(x,x.T) + lamda*np.eye(3))
        w_reg = np.dot(inv,np.dot(x,y.T))
    
        #compute E_val
        E_val = errorRate(w_reg, split_validate)
        #print("E_val:", E_val)
        accum_E_val += E_val
        
    E_cv = accum_E_val/v_fold
    print("E_cv:", E_cv)
    return E_cv

def start():
    arr = np.arange(13)-10
    print(arr)
    for i in arr:
        if(i<0): test = 1.0/np.power(10,-1*i) 
        else: test = np.power(10,i)
        print(test)
        ridgeReg(test)

start()

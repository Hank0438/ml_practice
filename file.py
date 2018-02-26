#目標: 讓每row前4個element為一個四維的x，第5個為一個一維的y

#simple file read
#f = open('pla_data.txt','r')
#print(f.read())


#panda file read
import pandas as pd
import os
import numpy as np

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
data = pd.read_csv(SCRIPT_PATH + "/data/pla_data.txt", header=None, encoding='utf-8', delim_whitespace=True)
dataset = np.array(data)
print(len(dataset[0]))

print(dataset)

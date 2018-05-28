import pandas as pd 
import numpy as np 

r1=pd.read_csv("test.csv",header=None)
r2=pd.read_csv("test1.csv",header=None)
r3=pd.read_csv("test2.csv",header=None)


r1_array=r1.values[:,2]
r2_array=r2.values[:,2]
r3_array=r3.values[:,2]

print(r1_array.shape)
print(r2_array.shape)
print(r3_array.shape)

r=np.vstack((r1_array.reshape(-1,1),r2_array.reshape(-1,1),r3_array.reshape(-1,1)))

print(r.shape)


result=pd.DataFrame({"pro":r[:,0]})

result.to_csv("result.csv",index=False,sep=",",header=None)